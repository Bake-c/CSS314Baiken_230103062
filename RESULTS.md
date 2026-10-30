# Karp-Flatt Experiment Results

## Hardware

- Student ID: 230103062
- CPU model: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Logical processors: 12
- Operating system: Windows-10-10.0.26200-SP0
- Python version: 3.11.5

## Measurements

| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Diagnosis |
|---|---:|---:|---:|---:|---:|---|
| A: Serial tail | 3.941 | 2.176 | 1.416 | 0.1046 | 0.1458 | Fixed serial work |
| B: Lock contention | 1.208 | 10.336 | 11.044 | 16.1102 | 11.8545 | Shared lock contention |
| C: Privatized counter | 0.586 | 0.349 | 0.249 | 0.1921 | 0.2333 | Private accumulation and final reduction |

From the displayed, rounded runtimes:

- Experiment A speedup: S2 ≈ 1.81; S4 ≈ 2.78.
- Experiment C efficiency: E4 ≈ 58.8%.
