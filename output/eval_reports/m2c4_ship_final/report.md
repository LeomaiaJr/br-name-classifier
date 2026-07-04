# BR Name Classifier Eval Report

- Split: `final`
- Eval rows: `1839`
- Model hash: `7a577dbceaf4a5979b9faf8881446e911e44b7e7efa22331a53692519ef38103`
- Generated at: `20260704T100049836274Z`
- Benchmark: `130/130` passed

## Stage 1 Gate

| Country | n | >=15 | >=30 | >=50 | >=70 |
| --- | --- | --- | --- | --- | --- |
| AO | 150 | 64.0% | 62.7% | 47.3% | 41.3% |
| BR | 300 | 63.7% | 63.3% | 47.0% | 43.3% |
| CV | 150 | 74.0% | 71.3% | 53.3% | 50.7% |
| ES-MX-generic | 150 | 6.7% | 6.0% | 1.3% | 0.7% |
| GW | 150 | 54.0% | 52.0% | 41.3% | 38.0% |
| HT | 100 | 10.0% | 10.0% | 1.0% | 1.0% |
| MZ | 150 | 57.3% | 56.0% | 37.3% | 30.0% |
| PT | 300 | 63.3% | 62.3% | 47.7% | 46.3% |
| ST | 89 | 66.3% | 66.3% | 58.4% | 51.7% |
| TL | 150 | 76.7% | 75.3% | 60.7% | 58.0% |
| US | 150 | 2.0% | 1.3% | 0.0% | 0.0% |

## Negative False Positives

| Threshold | FP count | Negative total | FP rate |
| --- | --- | --- | --- |
| >=15 | 23 | 400 | 5.8% |
| >=30 | 21 | 400 | 5.2% |
| >=50 | 3 | 400 | 0.8% |
| >=70 | 2 | 400 | 0.5% |

## Stage 2 Country Attribution

- Evaluated rows: `1439`
- Operating point: `top1_top2_margin >= 0.45`
- Mean entropy: `2.0752`
- Mean normalized entropy: `0.8028`
- Mean top1-top2 margin: `0.2133`

### Attributed Precision / Coverage

| Class | Attributed | Correct | Actual total | Precision | Coverage |
| --- | --- | --- | --- | --- | --- |
| BR | 24 | 6 | 300 | 25.0% | 2.0% |
| PT | 0 | 0 | 300 | n/a | 0.0% |
| CV | 22 | 12 | 150 | 54.5% | 8.0% |
| AO | 0 | 0 | 150 | n/a | 0.0% |
| MZ | 14 | 14 | 150 | 100.0% | 9.3% |
| PALOP_OTHER | 0 | 0 | 389 | n/a | 0.0% |

### Top-1 Accuracy (All Rows, Diagnostic Only)

| Class | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| BR | 290 | 300 | 96.7% |
| PT | 10 | 300 | 3.3% |
| CV | 17 | 150 | 11.3% |
| AO | 1 | 150 | 0.7% |
| MZ | 23 | 150 | 15.3% |
| PALOP_OTHER | 3 | 389 | 0.8% |

### Confusion Matrix

| Actual | BR | PT | CV | AO | MZ | PALOP_OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| BR | 290 | 6 | 4 | 0 | 0 | 0 |
| PT | 287 | 10 | 1 | 0 | 0 | 2 |
| CV | 130 | 3 | 17 | 0 | 0 | 0 |
| AO | 140 | 6 | 1 | 1 | 1 | 1 |
| MZ | 117 | 6 | 1 | 1 | 23 | 2 |
| PALOP_OTHER | 364 | 16 | 5 | 0 | 1 | 3 |

### Calibration

| Bin | Range | n | Accuracy | Avg confidence | Abs gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0-0.1 | 0 | n/a | n/a | n/a |
| 1 | 0.1-0.2 | 0 | n/a | n/a | n/a |
| 2 | 0.2-0.3 | 8 | 75.0% | 27.9% | 47.1% |
| 3 | 0.3-0.4 | 53 | 18.9% | 35.1% | 16.2% |
| 4 | 0.4-0.5 | 1259 | 22.2% | 45.0% | 22.8% |
| 5 | 0.5-0.6 | 48 | 31.2% | 54.5% | 23.3% |
| 6 | 0.6-0.7 | 33 | 33.3% | 64.9% | 31.6% |
| 7 | 0.7-0.8 | 29 | 62.1% | 75.0% | 13.0% |
| 8 | 0.8-0.9 | 9 | 55.6% | 83.6% | 28.1% |
| 9 | 0.9-1.0 | 0 | n/a | n/a | n/a |
