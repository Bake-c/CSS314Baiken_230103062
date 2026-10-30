import time
from multiprocessing import Lock, Process, Value


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    # Распределяем остаток, чтобы сумма всегда была точной.
    base, remainder = divmod(total_increments, p)
    tasks = [base + (1 if i < remainder else 0) for i in range(p)]

    counter = Value("i", 0, lock=False)
    lock = Lock()

    processes = [
        Process(target=contended_worker, args=(counter, lock, iterations))
        for iterations in tasks
    ]

    start = time.perf_counter()

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker failed: exit code {process.exitcode}")

    elapsed = time.perf_counter() - start

    assert counter.value == total_increments, (
        f"Expected {total_increments}, got {counter.value}"
    )

    return elapsed


def print_results(runtimes):
    t1 = runtimes[1]

    print("\nExperiment B: Lock contention")
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
        print(f"Running experiment B with {p} process(es)...")
        runtimes[p] = run_contended(p)

    print_results(runtimes)