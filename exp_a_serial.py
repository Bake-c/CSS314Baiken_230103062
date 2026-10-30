import math
import time
from multiprocessing import Pool


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    start = time.perf_counter()

    
    time.sleep(0.40)

    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)

    return time.perf_counter() - start


def print_results(runtimes):
    t1 = runtimes[1]

    print("\nExperiment A: Serial tail")
    print("P | Time (s) | Speedup | Karp-Flatt e")
    print("-" * 43)

    for p, tp in runtimes.items():
        speedup = t1 / tp

        if p == 1:
            e_text = "-"
        else:
            e = ((p * tp / t1) - 1.0) / (p - 1.0)
            e_text = f"{e:.4f}"

        print(f"{p} | {tp:8.3f} | {speedup:7.3f} | {e_text}")


if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {}

    for p in cores:
        print(f"Running experiment A with {p} process(es)...")
        runtimes[p] = run_experiment(p)

    print_results(runtimes)