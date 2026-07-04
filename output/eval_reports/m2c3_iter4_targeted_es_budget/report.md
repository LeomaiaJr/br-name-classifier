# BR Name Classifier Eval Report

- Split: `tuning`
- Eval rows: `1838`
- Model hash: `ffe7cdbc4d63c6b0914ef63c85575c8bc2c3f1495e313b997740d024f8d95a14`
- Generated at: `20260704T095337084267Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 65.3% | 63.3% | 44.7% | 39.3% |
| BR | 300 | 70.3% | 70.0% | 53.0% | 50.0% |
| CV | 150 | 70.7% | 70.7% | 53.3% | 48.7% |
| ES-MX-generic | 150 | 4.0% | 2.7% | 0.7% | 0.7% |
| GW | 150 | 58.0% | 57.3% | 44.0% | 40.7% |
| HT | 100 | 3.0% | 3.0% | 1.0% | 1.0% |
| MZ | 150 | 55.3% | 54.7% | 40.0% | 35.3% |
| PT | 300 | 65.0% | 62.3% | 43.3% | 40.3% |
| ST | 88 | 73.9% | 70.5% | 55.7% | 55.7% |
| TL | 150 | 80.0% | 78.0% | 60.0% | 58.0% |
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
- Mean entropy: `2.0914`
- Mean normalized entropy: `0.8091`
- Mean top1-top2 margin: `0.2057`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 26 | 7 | 300 | 26.9% | 2.3% |
| PT | 0 | 0 | 300 | n/a | 0.0% |
| CV | 3 | 2 | 150 | 66.7% | 1.3% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 9 | 9 | 150 | 100.0% | 6.0% |
| PALOP_OTHER | 0 | 0 | 388 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 294 | 300 | 98.0% |
| PT | 13 | 300 | 4.3% |
| CV | 4 | 150 | 2.7% |
| AO | 2 | 150 | 1.3% |
| MZ | 16 | 150 | 10.7% |
| PALOP_OTHER | 6 | 388 | 1.5% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 294 | 4 | 0 | 0 | 0 | 2 |
| PT | 282 | 13 | 0 | 2 | 0 | 3 |
| CV | 144 | 2 | 4 | 0 | 0 | 0 |
| AO | 143 | 5 | 0 | 2 | 0 | 0 |
| MZ | 123 | 11 | 0 | 0 | 16 | 0 |
| PALOP_OTHER | 369 | 12 | 1 | 0 | 0 | 6 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 13 | 23.1% | 27.3% | 4.3% |
| 3 | 0.3-0.4 | 73 | 21.9% | 35.0% | 13.1% |
| 4 | 0.4-0.5 | 1253 | 22.7% | 45.0% | 22.3% |
| 5 | 0.5-0.6 | 56 | 23.2% | 55.2% | 31.9% |
| 6 | 0.6-0.7 | 22 | 22.7% | 64.3% | 41.6% |
| 7 | 0.7-0.8 | 16 | 68.8% | 74.9% | 6.2% |
| 8 | 0.8-0.9 | 5 | 40.0% | 81.9% | 41.9% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
