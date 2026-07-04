# BR Name Classifier Eval Report

- Split: `tuning`
- Eval rows: `1838`
- Model hash: `f559127d1bc4bc966bc91e3745858ef24662cfafa9ffabc9089fe0e28a0ce4da`
- Generated at: `20260704T090212994661Z`
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
| TL | 150 | 84.0% | 81.3% | 63.3% | 63.3% |
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
- Mean entropy: `2.0849`
- Mean normalized entropy: `0.8065`
- Mean top1-top2 margin: `0.2037`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 14 | 5 | 300 | 35.7% | 1.7% |
| PT | 4 | 3 | 300 | 75.0% | 1.0% |
| CV | 20 | 9 | 150 | 45.0% | 6.0% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 12 | 12 | 150 | 100.0% | 8.0% |
| PALOP_OTHER | 0 | 0 | 388 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 243 | 300 | 81.0% |
| PT | 48 | 300 | 16.0% |
| CV | 49 | 150 | 32.7% |
| AO | 7 | 150 | 4.7% |
| MZ | 31 | 150 | 20.7% |
| PALOP_OTHER | 15 | 388 | 3.9% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 243 | 20 | 21 | 7 | 3 | 6 |
| PT | 202 | 48 | 32 | 6 | 4 | 8 |
| CV | 87 | 9 | 49 | 2 | 2 | 1 |
| AO | 114 | 11 | 12 | 7 | 6 | 0 |
| MZ | 94 | 17 | 4 | 3 | 31 | 1 |
| PALOP_OTHER | 257 | 36 | 51 | 21 | 8 | 15 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 85 | 17.6% | 26.9% | 9.3% |
| 3 | 0.3-0.4 | 183 | 18.6% | 35.4% | 16.8% |
| 4 | 0.4-0.5 | 966 | 26.4% | 45.0% | 18.6% |
| 5 | 0.5-0.6 | 128 | 35.9% | 54.8% | 18.9% |
| 6 | 0.6-0.7 | 61 | 59.0% | 63.8% | 4.8% |
| 7 | 0.7-0.8 | 13 | 46.2% | 73.6% | 27.4% |
| 8 | 0.8-0.9 | 2 | 50.0% | 83.0% | 33.0% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
