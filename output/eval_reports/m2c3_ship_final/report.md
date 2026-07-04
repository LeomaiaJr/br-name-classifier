# BR Name Classifier Eval Report

- Split: `final`
- Eval rows: `1839`
- Model hash: `493bfb0b1963e45f4b53abdefd0bbea7f8fca86e4f130c8c93b8c69de0450766`
- Generated at: `20260704T095102810212Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 68.7% | 67.3% | 47.3% | 41.3% |
| BR | 300 | 70.7% | 70.0% | 47.0% | 43.3% |
| CV | 150 | 74.7% | 72.0% | 46.7% | 44.0% |
| ES-MX-generic | 150 | 15.3% | 12.7% | 1.3% | 0.7% |
| GW | 150 | 56.0% | 54.0% | 40.0% | 36.7% |
| HT | 100 | 10.0% | 10.0% | 1.0% | 1.0% |
| MZ | 150 | 63.3% | 62.0% | 35.3% | 26.7% |
| PT | 300 | 68.7% | 67.7% | 47.7% | 46.3% |
| ST | 89 | 74.2% | 74.2% | 57.3% | 50.6% |
| TL | 150 | 83.3% | 81.3% | 60.7% | 58.0% |
| US | 150 | 2.0% | 1.3% | 0.0% | 0.0% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 36 | 400 | 9.0% |
| >=30 | 31 | 400 | 7.8% |
| >=50 | 3 | 400 | 0.8% |
| >=70 | 2 | 400 | 0.5% |

## Stage 2 Country Attribution

- Evaluated rows: `1439`
- Operating point: `top1_top2_margin >= 0.45`
- Mean entropy: `2.0913`
- Mean normalized entropy: `0.8090`
- Mean top1-top2 margin: `0.2060`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 24 | 6 | 300 | 25.0% | 2.0% |
| PT | 0 | 0 | 300 | n/a | 0.0% |
| CV | 3 | 0 | 150 | 0.0% | 0.0% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 8 | 8 | 150 | 100.0% | 5.3% |
| PALOP_OTHER | 0 | 0 | 389 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 294 | 300 | 98.0% |
| PT | 10 | 300 | 3.3% |
| CV | 2 | 150 | 1.3% |
| AO | 1 | 150 | 0.7% |
| MZ | 17 | 150 | 11.3% |
| PALOP_OTHER | 3 | 389 | 0.8% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 294 | 6 | 0 | 0 | 0 | 0 |
| PT | 288 | 10 | 0 | 0 | 0 | 2 |
| CV | 144 | 4 | 2 | 0 | 0 | 0 |
| AO | 141 | 6 | 0 | 1 | 1 | 1 |
| MZ | 122 | 7 | 1 | 1 | 17 | 2 |
| PALOP_OTHER | 367 | 16 | 2 | 0 | 1 | 3 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 8 | 75.0% | 27.9% | 47.1% |
| 3 | 0.3-0.4 | 55 | 18.2% | 35.2% | 17.0% |
| 4 | 0.4-0.5 | 1287 | 22.0% | 45.0% | 23.0% |
| 5 | 0.5-0.6 | 46 | 30.4% | 54.5% | 24.0% |
| 6 | 0.6-0.7 | 23 | 13.0% | 64.6% | 51.5% |
| 7 | 0.7-0.8 | 11 | 54.5% | 75.5% | 21.0% |
| 8 | 0.8-0.9 | 9 | 55.6% | 83.6% | 28.1% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
