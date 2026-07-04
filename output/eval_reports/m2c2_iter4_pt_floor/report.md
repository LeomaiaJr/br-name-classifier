# BR Name Classifier Eval Report

- Split: `tuning`
- Eval rows: `1838`
- Model hash: `07b4fcfb6631b669e8a9a9f4186f184008bc1355345113b0b2690c59a323719b`
- Generated at: `20260704T090126641221Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 68.7% | 66.7% | 49.3% | 48.7% |
| BR | 300 | 77.3% | 77.0% | 61.0% | 60.7% |
| CV | 150 | 78.7% | 77.3% | 63.3% | 63.3% |
| ES-MX-generic | 150 | 10.0% | 8.0% | 1.3% | 1.3% |
| GW | 150 | 64.7% | 63.3% | 48.7% | 48.7% |
| HT | 100 | 7.0% | 7.0% | 1.0% | 1.0% |
| MZ | 150 | 62.0% | 61.3% | 44.0% | 44.0% |
| PT | 300 | 70.3% | 67.0% | 51.0% | 50.3% |
| ST | 88 | 80.7% | 78.4% | 55.7% | 55.7% |
| TL | 150 | 84.0% | 81.3% | 64.0% | 63.3% |
| US | 150 | 3.3% | 3.3% | 0.0% | 0.0% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 27 | 400 | 6.8% |
| >=30 | 24 | 400 | 6.0% |
| >=50 | 3 | 400 | 0.8% |
| >=70 | 3 | 400 | 0.8% |

## Stage 2 Country Attribution

- Evaluated rows: `1438`
- Operating point: `top1_top2_margin >= 0.45`
- Mean entropy: `2.0571`
- Mean normalized entropy: `0.7958`
- Mean top1-top2 margin: `0.2048`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 116 | 56 | 300 | 48.3% | 18.7% |
| PT | 4 | 3 | 300 | 75.0% | 1.0% |
| CV | 0 | 0 | 150 | n/a | 0.0% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 0 | 0 | 150 | n/a | 0.0% |
| PALOP_OTHER | 0 | 0 | 388 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 246 | 300 | 82.0% |
| PT | 65 | 300 | 21.7% |
| CV | 16 | 150 | 10.7% |
| AO | 8 | 150 | 5.3% |
| MZ | 30 | 150 | 20.0% |
| PALOP_OTHER | 17 | 388 | 4.4% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 246 | 27 | 9 | 9 | 3 | 6 |
| PT | 207 | 65 | 7 | 9 | 4 | 8 |
| CV | 109 | 16 | 16 | 4 | 2 | 3 |
| AO | 120 | 15 | 1 | 8 | 6 | 0 |
| MZ | 93 | 20 | 2 | 4 | 30 | 1 |
| PALOP_OTHER | 283 | 40 | 12 | 27 | 9 | 17 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 149 | 18.1% | 27.0% | 8.9% |
| 3 | 0.3-0.4 | 321 | 20.6% | 35.2% | 14.6% |
| 4 | 0.4-0.5 | 658 | 23.3% | 44.9% | 21.7% |
| 5 | 0.5-0.6 | 164 | 39.0% | 55.7% | 16.6% |
| 6 | 0.6-0.7 | 97 | 46.4% | 64.2% | 17.8% |
| 7 | 0.7-0.8 | 45 | 53.3% | 74.1% | 20.8% |
| 8 | 0.8-0.9 | 4 | 75.0% | 82.2% | 7.2% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
