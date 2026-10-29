import csv
import statistics
import time
from pathlib import Path
import numpy as np
from salt import read_params

ROOT = Path(__file__).resolve().parents[1]


def measure(data, stride):
    start = time.perf_counter_ns()
    total = float(np.sum(data[::stride], dtype=np.float64))
    return time.perf_counter_ns() - start, total


def main():
    params = read_params()
    data = np.ones(params['buffer_elements'], dtype=np.float64)
    output = ROOT / 'results' / 'stride_results.tsv'
    rows = []
    for s in (1, 2, 4, 8, 16, 32, 64, 128):
        touched = len(data[::s])
        for _ in range(5):
            _, total = measure(data, s)
            if total != touched:
                raise RuntimeError(f'Incorrect stride reduction for stride={s}')
        samples = [measure(data, s) for _ in range(5)]
        if any(total != touched for _, total in samples):
            raise RuntimeError(f'Incorrect stride reduction for stride={s}')
        timings = [ns for ns, _ in samples]
        median_ns = statistics.median(timings)
        rows.append({'stride': s, 'elements_touched': touched,
                     'median_time_ns': median_ns, 'ns_per_element': median_ns / touched,
                     'latency_ns': median_ns / touched, 'reduction_sum': samples[-1][1]})
        print(f'stride={s:3d} median={median_ns/1e6:.3f} ms  ns/element={median_ns/touched:.3f}')
    with output.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter='\t')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
