import json
import statistics
import time
from pathlib import Path
import numpy as np
from salt import read_params

ROOT = Path(__file__).resolve().parents[1]


def naive_ijk(a, b):
    m = len(a)
    c = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            c[i, j] = np.sum(a[i, :] * b[:, j], dtype=np.float64)
    return c


def interchanged_ikj(a, b):
    m = len(a)
    c = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for k in range(m):
            aik = a[i, k]
            c[i, :] += aik * b[k, :]
    return c


def time_runs(fn, a, b):
    times = []
    output = None
    for _ in range(3):
        start = time.perf_counter_ns()
        output = fn(a, b)
        times.append(time.perf_counter_ns() - start)
    return statistics.median(times), output


def main():
    p = read_params()
    m = p['N'] // 2
    rng = np.random.default_rng(p['seed'])
    a = rng.random((m, m), dtype=np.float64)
    b = rng.random((m, m), dtype=np.float64)
    t_naive, naive_result = time_runs(naive_ijk, a, b)
    t_inter, inter_result = time_runs(interchanged_ikj, a, b)
    max_error = float(np.max(np.abs(naive_result - inter_result)))
    close = bool(np.allclose(naive_result, inter_result, atol=1e-12, rtol=0))
    if not close:
        raise RuntimeError(f'Matrix computations failed absolute tolerance 1e-12: {max_error}')
    out = {'student_id': p['student_id'], 'N': p['N'], 'M': m,
           'note': 'NumPy inner reductions/row operations implement logical i-j-k and i-k-j traversal; not pure-scalar loops',
           'naive_median_ns': t_naive, 'interchanged_median_ns': t_inter,
           'naive_time_s': t_naive / 1e9, 'interchanged_time_s': t_inter / 1e9,
           'speedup': t_naive / t_inter, 'max_absolute_error': max_error,
           'matches_tolerance_1e_12': close}
    (ROOT / 'results' / 'loop_interchange.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
