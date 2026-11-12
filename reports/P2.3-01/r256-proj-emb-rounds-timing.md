# P2.3-01 round timing

Records: 188; skipped lines: 0; files: `runs/P2.3-01/run_logs/r256-proj-emb-rounds/timing/timing-main.jsonl`, `runs/P2.3-01/run_logs/r256-proj-emb-rounds/timing/timing-rank0.jsonl`, `runs/P2.3-01/run_logs/r256-proj-emb-rounds/timing/timing-rank1.jsonl`, `runs/P2.3-01/run_logs/r256-proj-emb-rounds/timing/timing-rank2.jsonl`, `runs/P2.3-01/run_logs/r256-proj-emb-rounds/timing/timing-rank3.jsonl`.

## Per puzzle

| Puzzle | Worker | Train s | Decode s | Selection s | Overhead s | Total s | Train tokens | Decode tokens | Selection tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0934a4d8 | rank3 | 288.17 | 34.78 | 107.80 | 212.39 | 643.14 | 371328 | 61392 | 495360 |
| 36a08778 | rank1 | 162.76 | 879.13 | 87.80 | 1496.93 | 2626.62 | 200064 | 69440 | 409552 |
| 981571dc | rank2 | 588.92 | 378.24 | 13.94 | 1414.62 | 2395.73 | 716544 | 104544 | 59712 |
| aa4ec2a5 | rank0 | 250.04 | 456.04 | 40.98 | 525.61 | 1272.69 | 326272 | 52536 | 192576 |
| mean of 4 |  | 322.47 | 437.05 | 62.63 | 912.39 | 1734.54 | 403552 | 71978 | 289300 |
| sum |  | 1289.89 | 1748.19 | 250.53 | 3649.56 | 6938.18 | 1614208 | 287912 | 1157200 |

Overhead: the `puzzle` record minus train, decode, and selection; for a puzzle without a `puzzle` record, the sum of its other ops. Token means run over the puzzles that report a count.

## Ops outside puzzles

| Op | Records | Total s | Mean s |
|---|---:|---:|---:|
| load | 4 | 387.79 | 96.95 |
| main_selection | 1 | 0.10 | 0.10 |

## Cost model: seconds = a + b * tokens

| Op | n | a (s) | b (s per token) | R^2 |
|---|---:|---:|---:|---:|
| candidate_train | 2 | -9.1500 | 2.2875e-04 | 1.0000 |
| decode | 20 | 95.0372 | -5.2985e-04 | 0.0027 |
| final_decode | 12 | 621.2934 | -3.2236e-02 | 0.7886 |
| final_selection | 26 | 2.0285 | 1.6482e-04 | 0.8390 |
| fingerprint | 0 | n/a | n/a | n/a |
| gate | 0 | n/a | n/a | n/a |
| gate_decode | 24 | 4.6202 | 3.0761e-03 | 0.5017 |
| gate_selection | 19 | -0.6110 | 2.4410e-04 | 0.9982 |
| load | 0 | n/a | n/a | n/a |
| main_selection | 0 | n/a | n/a | n/a |
| puzzle | 0 | n/a | n/a | n/a |
| redecode | 0 | n/a | n/a | n/a |
| redecode_decode | 8 | 25.6366 | 2.6508e-03 | 0.8598 |
| redecode_selection | 7 | -0.5942 | 2.4732e-04 | 0.9999 |
| round | 0 | n/a | n/a | n/a |
| selection | 41 | -0.5233 | 2.3504e-04 | 0.9957 |
| snapshot | 0 | n/a | n/a | n/a |
| train | 4 | -15.5109 | 8.3752e-04 | 0.9976 |

Least squares over the op's records that carry a token count (n); a alone is the mean when the token counts do not vary.
