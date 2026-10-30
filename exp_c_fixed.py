import time
from multiprocessing import Pool


def private_worker(iterations):
    local_sum = 0

    for _ in range(iterations):
        local_sum += 1

    return local_sum


def run_privatized(p, total_increments=20_000_000):
    base, remainder = divmod(total_increments, p)
    tasks = [base + (1 if i < remainder else 0) for i in range(p)]

    start = time.perf_counter()

    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)

    total = sum(results)
    elapsed = time.perf_counter() - start

    assert total == total_increments, (
        f"Expected {total_increments}, got {total}"
    )

    return elapsed


def print_results(runtimes):
    t1 = runtimes[1]

    print("\nExperiment C: Privatized counter")
    print("P | Time (s) | Speedup | Efficiency | Karp-Flatt e")
    print("-" * 58)

    for p, tp in runtimes.items():
        speedup = t1 / tp
        efficiency = speedup / p * 100

        if p == 1:
            e_text = "-"
        else:
            e = ((p * tp / t1) - 1.0) / (p - 1.0)
            e_text = f"{e:.4f}"

        print(
            f"{p} | {tp:8.3f} | {speedup:7.3f} | "
            f"{efficiency:9.1f}% | {e_text}"
        )


if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {}

    for p in cores:
        print(f"Running experiment C with {p} process(es)...")
        runtimes[p] = run_privatized(p)

    print_results(runtimes)