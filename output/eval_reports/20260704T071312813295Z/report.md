# BR Name Classifier Eval Report

- Split: `tuning`
- Eval rows: `1738`
- Model hash: `76c08b802a96d8e80c7d726316182639a03fdbe7e7535472681243b05716d1ce`
- Generated at: `20260704T071312813295Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 60.7% | 58.0% | 42.0% | 40.7% |
| BR | 300 | 72.0% | 71.0% | 56.0% | 51.7% |
| CV | 150 | 70.0% | 68.7% | 53.3% | 52.0% |
| ES-MX-generic | 150 | 4.0% | 3.3% | 0.7% | 0.7% |
| GW | 150 | 57.3% | 55.3% | 42.0% | 41.3% |
| MZ | 150 | 48.7% | 46.7% | 30.7% | 28.7% |
| PT | 300 | 67.3% | 63.0% | 48.3% | 47.0% |
| ST | 88 | 75.0% | 69.3% | 55.7% | 55.7% |
| TL | 150 | 80.0% | 75.3% | 59.3% | 58.0% |
| US | 150 | 3.3% | 3.3% | 0.0% | 0.0% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 11 | 300 | 3.7% |
| >=30 | 10 | 300 | 3.3% |
| >=50 | 1 | 300 | 0.3% |
| >=70 | 1 | 300 | 0.3% |

## Stage 2 Country Attribution

STAGE2: N/A (model does not attribute countries yet)
