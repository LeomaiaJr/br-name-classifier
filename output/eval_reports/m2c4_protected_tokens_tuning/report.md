# BR Name Classifier Eval Report

- Split: `tuning`
- Eval rows: `1838`
- Model hash: `7a577dbceaf4a5979b9faf8881446e911e44b7e7efa22331a53692519ef38103`
- Generated at: `20260704T100027114146Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 65.3% | 63.3% | 44.7% | 39.3% |
| BR | 300 | 71.0% | 70.7% | 53.7% | 50.7% |
| CV | 150 | 72.7% | 72.7% | 57.3% | 52.7% |
| ES-MX-generic | 150 | 4.0% | 2.7% | 0.7% | 0.7% |
| GW | 150 | 58.0% | 57.3% | 45.3% | 42.0% |
| HT | 100 | 3.0% | 3.0% | 1.0% | 1.0% |
| MZ | 150 | 57.3% | 56.7% | 42.0% | 37.3% |
| PT | 300 | 65.0% | 62.3% | 43.7% | 40.7% |
| ST | 88 | 73.9% | 70.5% | 55.7% | 55.7% |
| TL | 150 | 80.7% | 78.0% | 60.0% | 58.0% |
| US | 150 | 3.3% | 3.3% | 0.0% | 0.0% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 14 | 400 | 3.5% |
| >=30 | 12 | 400 | 3.0% |
| >=50 | 2 | 400 | 0.5% |
| >=70 | 2 | 400 | 0.5% |

## Stage 2 Country Attribution

- Evaluated rows: `1438`
- Operating point: `top1_top2_margin >= 0.45`
- Mean entropy: `2.0802`
- Mean normalized entropy: `0.8047`
- Mean top1-top2 margin: `0.2104`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 26 | 7 | 300 | 26.9% | 2.3% |
| PT | 0 | 0 | 300 | n/a | 0.0% |
| CV | 14 | 7 | 150 | 50.0% | 4.7% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 14 | 14 | 150 | 100.0% | 9.3% |
| PALOP_OTHER | 0 | 0 | 388 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 291 | 300 | 97.0% |
| PT | 13 | 300 | 4.3% |
| CV | 14 | 150 | 9.3% |
| AO | 2 | 150 | 1.3% |
| MZ | 21 | 150 | 14.0% |
| PALOP_OTHER | 6 | 388 | 1.5% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 291 | 4 | 3 | 0 | 0 | 2 |
| PT | 280 | 13 | 2 | 2 | 0 | 3 |
| CV | 135 | 1 | 14 | 0 | 0 | 0 |
| AO | 143 | 5 | 0 | 2 | 0 | 0 |
| MZ | 118 | 11 | 0 | 0 | 21 | 0 |
| PALOP_OTHER | 366 | 12 | 4 | 0 | 0 | 6 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 13 | 23.1% | 27.3% | 4.3% |
| 3 | 0.3-0.4 | 72 | 22.2% | 35.1% | 12.8% |
| 4 | 0.4-0.5 | 1232 | 22.9% | 45.1% | 22.2% |
| 5 | 0.5-0.6 | 59 | 25.4% | 55.2% | 29.8% |
| 6 | 0.6-0.7 | 29 | 34.5% | 64.4% | 29.9% |
| 7 | 0.7-0.8 | 28 | 67.9% | 74.7% | 6.8% |
| 8 | 0.8-0.9 | 5 | 40.0% | 81.9% | 41.9% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
