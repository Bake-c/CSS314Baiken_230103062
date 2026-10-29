import argparse
import csv
import ctypes
import multiprocessing as mp
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(array, worker_id, iterations, mode):
    if mode == 'privatized':
        local = 0
        for _ in range(iterations):
            local += 1
        array[worker_id * 8] = local
    else:
        index = worker_id if mode == 'unpadded' else worker_id * 8
        for _ in range(iterations):
            array[index] += 1


def benchmark(p, iterations, mode):
    if p < 1 or iterations < 1 or mode not in ('unpadded', 'padded', 'privatized'):
        raise ValueError('Invalid counter benchmark configuration')
    count = p if mode == 'unpadded' else p * 8
    array = mp.RawArray(ctypes.c_int64, count)
    processes = [mp.Process(target=worker, args=(array, i, iterations, mode)) for i in range(p)]
    start = time.perf_counter_ns()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    duration = time.perf_counter_ns() - start
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError('One or more workers failed')
    total = sum(array)
    if total != p * iterations:
        raise RuntimeError(f'Wrong counter total: {total}, expected {p * iterations}')
    return duration, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=5_000_000)
    ap.add_argument('--runs', type=int, default=1)
    args = ap.parse_args()
    if args.iterations <= 0 or args.runs <= 0:
        ap.error('iterations and runs must be positive')
    data = {}
    totals = {}
    for mode in ('unpadded', 'padded', 'privatized'):
        data[mode] = {}
        for p in (1, 2, 4, 8):
            samples = [benchmark(p, args.iterations, mode) for _ in range(args.runs)]
            ns = statistics.median(t for t, _ in samples)
            data[mode][p] = ns
            totals[mode, p] = samples[-1][1]
            print(f'{mode:11s} P={p} median={ns/1e9:.3f}s')
    rows = []
    # One row per process count: the instructor verifier expects wide columns.
    for p in (1, 2, 4, 8):
        row = {'p': p, 'iterations_per_process': args.iterations, 'runs': args.runs}
        for mode in data:
            ns = data[mode][p]
            speedup = data[mode][1] / ns
            e = ((1 / speedup) - (1 / p)) / (1 - 1 / p) if p > 1 else ''
            row.update({f'{mode}_median_ns': ns, f'total_counts_{mode}': totals[mode, p],
                        f'speedup_{mode}': speedup, f'e_{mode}': e})
        rows.append(row)
    with (ROOT / 'results' / 'false_sharing_karp_flatt.tsv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter='\t')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    mp.freeze_support()
    main()
