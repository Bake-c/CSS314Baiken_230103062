# CSS314 Parallel Computing Practicum

On the Intel Core i5-12450H, stride-8 access took 3.16 times as long per element as stride-1, and loop interchange accelerated the matrix kernel by 2.34 times. At eight processes, unpadded counters took 2.163 s, padded counters 1.571 s, and privatized counters 0.658 s. Padding and privatization reduced elapsed time, although process overhead and the increasing total work still limited scaling. The corrected stencil shares updated boundary rows at every iteration and matched the serial reference at all tested process counts. Strong-scaling speedup reached about 1.13 at eight processes. The estimated Roofline is memory-bound: arithmetic intensity is 5/48 = 0.10417 FLOPs/byte, measured copy throughput is 27.94 GB/s, and measured NumPy GEMM throughput is 155.86 GFLOP/s. These are local measurements and an approximate traffic model, not theoretical hardware peaks.

## Reproduce the results

Run these commands from the directory containing `src/`, `results/`, and the instructor's `verify_outputs.py`:

```powershell
py -m pip install -r requirements.txt
py src/profile_env.py --student-id 30103062
py src/stride_bench.py
py src/dense_kernel.py
py src/false_sharing.py --iterations 5000000 --runs 3
py src/scaling_study.py
py verify_outputs.py
```

The saved dataset uses Student ID `30103062`, salted N = 1344, and seed = 17564. If the Student ID changes, rerun every benchmark before verification. Results are regenerated from actual executions; timings will vary.

The instructor verifier is unchanged. Its successful run creates `results/verification_receipt.sig`. The scripts additionally check numerical outputs during measurement: stride reductions, matrix agreement within absolute tolerance 1e-12, exact counter totals, and every stencil element against the serial reference.

## Implementation and measurement notes

- Windows cache sizes and cache-line size come from `GetLogicalProcessorInformationEx`. L1/L2 sizes are per cache instance; multiple values describe the CPU's heterogeneous cores.
- The matrix kernels preserve the outer i-j-k and i-k-j traversal using NumPy inner reductions and row operations. They are not pure scalar Python triple loops.
- Stride latency is saved in nanoseconds per touched element as `latency_ns`. Matrix median runtimes are saved in both nanoseconds and seconds. Counter data has one row per process count with separate columns for each mode, matching the instructor verifier's schema.
- Scaling uses 20 stencil iterations and the median of three runs. Two shared buffers and a synchronization point after each step ensure that neighboring partitions read the same iteration. Timings include process creation, shared-buffer setup, synchronization, and final output copying; numerical checks run outside the timer.
- The `speedup` column for weak scaling follows the formula printed in the assignment. `work_rate_speedup` and `weak_efficiency_percent` separately show throughput speedup and T1/Tp efficiency. Increasing per-worker work in the counter experiment means its Karp-Flatt values should not be interpreted as a fixed-work serial fraction.
- Roofline defaults to measured copy and GEMM throughput, with raw timing samples stored in `roofline_summary.json`. The 48-byte stencil traffic estimate excludes cache reuse, write allocation, and NumPy temporaries. Process and synchronization overhead can dominate actual runtime even when this model predicts memory-bound execution. Documented hardware limits can instead be supplied with `--bandwidth-gbps` and `--peak-gflops`.

## Regression tests

```powershell
py -m unittest discover -s tests -v
```

Tests cover matrix correctness, stencil boundaries across uneven partitions, odd/even iteration counts, excess workers, and Roofline classification. Git milestone tags and the final repository push are separate submission steps.
