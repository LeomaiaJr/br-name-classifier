# BR Name Classifier Eval Report

- Split: `all`
- Eval rows: `3477`
- Model hash: `76c08b802a96d8e80c7d726316182639a03fdbe7e7535472681243b05716d1ce`
- Generated at: `20260704T071305639884Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 300 | 60.3% | 58.0% | 43.3% | 40.3% |
| BR | 600 | 69.0% | 67.8% | 51.0% | 48.5% |
| CV | 300 | 70.0% | 67.0% | 51.0% | 49.7% |
| ES-MX-generic | 300 | 6.7% | 5.7% | 1.7% | 1.3% |
| GW | 300 | 53.3% | 51.0% | 40.7% | 40.0% |
| MZ | 300 | 46.0% | 42.7% | 27.7% | 25.3% |
| PT | 600 | 66.0% | 62.8% | 49.0% | 47.3% |
| ST | 177 | 71.2% | 67.8% | 54.8% | 52.0% |
| TL | 300 | 78.3% | 74.7% | 58.7% | 56.7% |
| US | 300 | 3.0% | 3.0% | 0.7% | 0.7% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 29 | 600 | 4.8% |
| >=30 | 26 | 600 | 4.3% |
| >=50 | 7 | 600 | 1.2% |
| >=70 | 6 | 600 | 1.0% |

## Stage 2 Country Attribution

STAGE2: N/A (model does not attribute countries yet)
