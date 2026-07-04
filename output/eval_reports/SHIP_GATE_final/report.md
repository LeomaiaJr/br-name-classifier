# BR Name Classifier Eval Report

- Split: `final`
- Eval rows: `1839`
- Model hash: `f559127d1bc4bc966bc91e3745858ef24662cfafa9ffabc9089fe0e28a0ce4da`
- Generated at: `20260704T092718178054Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 70.0% | 68.0% | 52.0% | 50.7% |
| BR | 300 | 71.3% | 70.3% | 51.3% | 50.7% |
| CV | 150 | 82.0% | 78.7% | 60.0% | 59.3% |
| ES-MX-generic | 150 | 16.0% | 12.7% | 2.7% | 2.7% |
| GW | 150 | 57.3% | 55.3% | 45.3% | 44.7% |
| HT | 100 | 15.0% | 15.0% | 2.0% | 2.0% |
| MZ | 150 | 64.0% | 62.7% | 40.7% | 39.3% |
| PT | 300 | 70.3% | 69.0% | 51.3% | 50.7% |
| ST | 89 | 76.4% | 76.4% | 62.9% | 61.8% |
| TL | 150 | 83.3% | 81.3% | 64.7% | 64.0% |
| US | 150 | 2.0% | 1.3% | 0.0% | 0.0% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 42 | 400 | 10.5% |
| >=30 | 36 | 400 | 9.0% |
| >=50 | 6 | 400 | 1.5% |
| >=70 | 6 | 400 | 1.5% |

## Stage 2 Country Attribution

- Evaluated rows: `1439`
- Operating point: `top1_top2_margin >= 0.45`
- Mean entropy: `2.0740`
- Mean normalized entropy: `0.8023`
- Mean top1-top2 margin: `0.2067`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 14 | 5 | 300 | 35.7% | 1.7% |
| PT | 4 | 1 | 300 | 25.0% | 0.3% |
| CV | 36 | 15 | 150 | 41.7% | 10.0% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 13 | 13 | 150 | 100.0% | 8.7% |
| PALOP_OTHER | 0 | 0 | 389 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 247 | 300 | 82.3% |
| PT | 41 | 300 | 13.7% |
| CV | 51 | 150 | 34.0% |
| AO | 3 | 150 | 2.0% |
| MZ | 25 | 150 | 16.7% |
| PALOP_OTHER | 11 | 389 | 2.8% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 247 | 18 | 24 | 5 | 2 | 4 |
| PT | 205 | 41 | 39 | 7 | 4 | 4 |
| CV | 87 | 7 | 51 | 1 | 4 | 0 |
| AO | 119 | 13 | 11 | 3 | 4 | 0 |
| MZ | 99 | 17 | 5 | 4 | 25 | 0 |
| PALOP_OTHER | 273 | 33 | 50 | 15 | 7 | 11 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 73 | 26.0% | 26.8% | 0.8% |
| 3 | 0.3-0.4 | 177 | 16.9% | 34.6% | 17.7% |
| 4 | 0.4-0.5 | 967 | 26.3% | 45.0% | 18.7% |
| 5 | 0.5-0.6 | 120 | 24.2% | 54.2% | 30.0% |
| 6 | 0.6-0.7 | 76 | 34.2% | 63.6% | 29.3% |
| 7 | 0.7-0.8 | 22 | 77.3% | 73.0% | 4.2% |
| 8 | 0.8-0.9 | 4 | 75.0% | 84.5% | 9.5% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
