import argparse
import csv
import ctypes
import json
import math
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np
from salt import read_params

ROOT = Path(__file__).resolve().parents[1]
_buffers = None


def step_chunk(args):
    """Serial reference with fixed outer boundary values."""
    block, iterations = args
    x = block.copy()
    for _ in range(iterations):
        y = x.copy()
        y[1:-1, 1:-1] = (x[1:-1, 1:-1] + x[:-2, 1:-1] + x[2:, 1:-1]
                         + x[1:-1, :-2] + x[1:-1, 2:]) / 5.0
        x = y
    return x[1:-1, 1:-1]


def init_buffers(first, second, shape):
    global _buffers
    _buffers = (np.frombuffer(first, dtype=np.float64).reshape(shape),
                np.frombuffer(second, dtype=np.float64).reshape(shape))


def update_rows(job):
    first, last, source = job
    x, y = _buffers[source], _buffers[1 - source]
    y[first:last, 1:-1] = (x[first:last, 1:-1] + x[first - 1:last - 1, 1:-1]
                          + x[first + 1:last + 1, 1:-1] + x[first:last, :-2]
                          + x[first:last, 2:]) / 5.0


def kernel(matrix, p, iterations):
    if matrix.ndim != 2 or min(matrix.shape) < 3 or p < 1 or iterations < 1:
        raise ValueError('Expected a matrix >= 3x3, positive processes and iterations')
    if p == 1:
        result = matrix.copy()
        result[1:-1, 1:-1] = step_chunk((matrix, iterations))
        return result
    # Both buffers retain the same fixed global boundary. Workers write disjoint
    # rows; map() completes all rows before swapping buffers for the next step.
    ctx = mp.get_context('spawn')
    first = ctx.RawArray(ctypes.c_double, int(matrix.size))
    second = ctx.RawArray(ctypes.c_double, int(matrix.size))
    views = [np.frombuffer(buffer, dtype=np.float64).reshape(matrix.shape)
             for buffer in (first, second)]
    for view in views:
        view[:] = matrix
    parts = np.array_split(np.arange(1, matrix.shape[0] - 1), min(p, matrix.shape[0] - 2))
    bounds = [(int(part[0]), int(part[-1]) + 1) for part in parts]
    with ctx.Pool(processes=len(bounds), initializer=init_buffers,
                  initargs=(first, second, matrix.shape)) as pool:
        for iteration in range(iterations):
            pool.map(update_rows, [(lo, hi, iteration % 2) for lo, hi in bounds])
    return views[iterations % 2].copy()


def timed(matrix, p, iterations, runs, reference):
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = kernel(matrix, p, iterations)
        samples.append(time.perf_counter_ns() - start)
        # Validation is outside the timed region and compares every element.
        if not np.allclose(output, reference, atol=1e-12, rtol=0):
            raise RuntimeError(f'Stencil mismatch at P={p}')
    return statistics.median(samples), float(output.sum(dtype=np.float64))


def measure_roofline_limits(seed):
    """Empirical NumPy copy/GEMM ceilings, not theoretical CPU maxima."""
    source = np.ones(16_777_216, dtype=np.float64)
    target = np.empty_like(source)
    copy_samples = []
    for run in range(7):
        start = time.perf_counter_ns()
        np.copyto(target, source)
        elapsed = time.perf_counter_ns() - start
        if run >= 2:
            copy_samples.append(elapsed)
    if not np.array_equal(source, target):
        raise RuntimeError('Bandwidth benchmark copy failed')
    byte_count = 2 * source.nbytes  # One read plus one write; write allocation excluded.
    bandwidth = byte_count / statistics.median(copy_samples)
    del source, target
    rng = np.random.default_rng(seed)
    n = 768
    a, b = rng.random((n, n)), rng.random((n, n))
    c = np.empty((n, n), dtype=np.float64)
    gemm_samples = []
    for run in range(7):
        start = time.perf_counter_ns()
        np.matmul(a, b, out=c)
        elapsed = time.perf_counter_ns() - start
        if run >= 2:
            gemm_samples.append(elapsed)
    if not np.all(np.isfinite(c)):
        raise RuntimeError('Compute benchmark produced non-finite values')
    compute = (2 * n ** 3) / statistics.median(gemm_samples)
    return bandwidth, compute, {
        'bandwidth_method': 'Median of 5 np.copyto runs after 2 warmups, 128 MiB source; bytes=read+write',
        'compute_method': 'Median of 5 float64 NumPy GEMM runs after 2 warmups; 768x768; FLOPs=2*N^3',
        'copy_samples_ns': copy_samples, 'gemm_samples_ns': gemm_samples,
        'caveat': 'Empirical library ceilings, not theoretical hardware peaks. Copy ignores write-allocate traffic; GEMM uses the installed BLAS thread configuration.',
    }


def build_roofline(host, seed, bandwidth=None, peak=None):
    measurements = {}
    bandwidth_source = 'user supplied'
    compute_source = 'user supplied'
    if bandwidth is None or peak is None:
        measured_bw, measured_peak, measurements = measure_roofline_limits(seed)
        if bandwidth is None:
            bandwidth, bandwidth_source = measured_bw, 'measured NumPy copy throughput'
        if peak is None:
            peak, compute_source = measured_peak, 'measured NumPy GEMM throughput'
    if not all(math.isfinite(x) and x > 0 for x in (bandwidth, peak)):
        raise ValueError('Roofline limits must be positive finite numbers')
    # Algorithmic traffic model: 5 double loads and one double store per cell.
    # This is not a measurement of DRAM traffic or NumPy temporary-array traffic.
    flops, byte_count = 5, 48
    ai = flops / byte_count
    regime = 'memory-bound' if bandwidth * ai < peak else 'compute-bound'
    return {
        'stencil': '5-point average', 'arithmetic_intensity': ai,
        'arithmetic_intensity_flops_per_byte': ai,
        'estimated_flops_per_cell': flops, 'conservative_bytes_per_cell': byte_count,
        'estimation_caveat': '48-byte algorithmic traffic model excludes cache reuse, write allocation and NumPy temporary arrays. Classification describes this estimated Roofline; process/IPC overhead can dominate measured runtime.',
        'host_cache': host['cache'], 'bandwidth_gbps_input': bandwidth,
        'peak_gflops_input': peak, 'bandwidth_source': bandwidth_source,
        'compute_source': compute_source, 'measurements': measurements,
        'memory_roof_gflops': bandwidth * ai,
        'attainable_gflops': min(bandwidth * ai, peak),
        'regime': regime, 'classification': regime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--bandwidth-gbps', type=float, default=None,
                    help='Documented/measured bandwidth; otherwise measure copy throughput')
    ap.add_argument('--peak-gflops', type=float, default=None,
                    help='Documented/measured compute ceiling; otherwise measure GEMM throughput')
    ap.add_argument('--max-processes', type=int, default=None)
    args = ap.parse_args()
    if args.iterations < 1 or args.runs < 1:
        ap.error('iterations and runs must be positive')
    if args.max_processes is not None and args.max_processes < 1:
        ap.error('--max-processes must be positive')
    for value in (args.bandwidth_gbps, args.peak_gflops):
        if value is not None and (not math.isfinite(value) or value <= 0):
            ap.error('Roofline inputs must be positive finite numbers')
    params = read_params()
    host = json.loads((ROOT / 'results' / 'host_profile.json').read_text(encoding='utf-8'))
    physical = host.get('physical_cores') or host.get('logical_threads') or 1
    max_p = min(physical, args.max_processes or physical)
    counts = [1]
    while counts[-1] * 2 <= max_p:
        counts.append(counts[-1] * 2)
    n = params['N']
    rows = []
    for study in ('strong', 'weak'):
        baseline_ns = None
        for p in counts:
            dimension = n if study == 'strong' else math.ceil(n * math.sqrt(p))
            matrix = np.random.default_rng(params['seed']).random((dimension, dimension))
            reference = kernel(matrix, 1, args.iterations)
            ns, checksum = timed(matrix, p, args.iterations, args.runs, reference)
            if p == 1:
                baseline_ns = ns
            work_1, work_p = (n - 2) ** 2, (dimension - 2) ** 2
            # Retain the assignment's explicit weak-scaling formula.
            speedup = (baseline_ns / ns if study == 'strong' else
                       p * work_1 * baseline_ns / (work_p * ns))
            serial = ((1 / speedup - 1 / p) / (1 - 1 / p)
                      if study == 'strong' and p > 1 else None)
            rows.append({
                'study': study, 'processes': p, 'dimension': dimension,
                'iterations': args.iterations, 'runs': args.runs,
                'median_time_ns': ns, 'checksum': checksum,
                'reference_checksum': float(reference.sum(dtype=np.float64)),
                'matches_reference': True,
                'speedup': speedup, 'efficiency_percent': 100 * speedup / p,
                'work_rate_speedup': (work_p / work_1) * baseline_ns / ns,
                'weak_efficiency_percent': 100 * baseline_ns / ns if study == 'weak' else '',
                'serial_fraction': serial if serial is not None else '',
                'amdahl_max_speedup': 1 / serial if serial is not None and serial > 0 else '',
            })
            print(f'{study:6s} P={p}: {ns/1e9:.3f}s, speedup={speedup:.3f}, numerical check PASS', flush=True)
            del matrix, reference
    roof = build_roofline(host, params['seed'], args.bandwidth_gbps, args.peak_gflops)
    with (ROOT / 'results' / 'scaling_metrics.tsv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'results' / 'roofline_summary.json').write_text(json.dumps(roof, indent=2), encoding='utf-8')
    print(f"Roofline: {roof['regime']}, AI={roof['arithmetic_intensity']:.6f} FLOPs/byte")


if __name__ == '__main__':
    mp.freeze_support()
    main()
