# P0.3-01 revision 2: DEV4 on 4xL4

Records: 73; skipped lines: 0; files: `reports/P0.3-01/timing-main.jsonl`, `reports/P0.3-01/timing-rank0.jsonl`, `reports/P0.3-01/timing-rank1.jsonl`, `reports/P0.3-01/timing-rank2.jsonl`, `reports/P0.3-01/timing-rank3.jsonl`.

## Per puzzle

| Puzzle | Worker | Train s | Decode s | Selection s | Overhead s | Total s | Train tokens | Decode tokens | Selection tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0934a4d8 | rank1 | 394.47 | 37.23 | 109.91 | 7.32 | 548.93 | 493696 | 76688 | 501904 |
| 36a08778 | rank0 | 210.31 | 746.78 | 142.83 | 7.27 | 1107.19 | 266240 | 85984 | 652368 |
| 981571dc | rank2 | 852.78 | 371.37 | 14.32 | 8.44 | 1246.92 | 955392 | 104544 | 59712 |
| aa4ec2a5 | rank3 | 305.58 | 311.64 | 25.45 | 5.53 | 648.20 | 408320 | 62792 | 111672 |
| mean of 4 |  | 440.78 | 366.76 | 73.13 | 7.14 | 887.81 | 530912 | 82502 | 331414 |
| sum |  | 1763.14 | 1467.02 | 292.51 | 28.55 | 3551.22 | 2123648 | 330008 | 1325656 |

Overhead: the `puzzle` record minus train, decode, and selection; for a puzzle without a `puzzle` record, the sum of its other ops. Token means run over the puzzles that report a count.

## Ops outside puzzles

| Op | Records | Total s | Mean s |
|---|---:|---:|---:|
| load | 4 | 144.28 | 36.07 |
| main_selection | 1 | 0.12 | 0.12 |

## Cost model: seconds = a + b * tokens

| Op | n | a (s) | b (s per token) | R^2 |
|---|---:|---:|---:|---:|
| decode | 20 | 80.5933 | -4.3890e-04 | 0.0025 |
| load | 0 | n/a | n/a | n/a |
| main_selection | 0 | n/a | n/a | n/a |
| puzzle | 0 | n/a | n/a | n/a |
| selection | 40 | -0.6290 | 2.3963e-04 | 0.9914 |
| train | 4 | -65.3107 | 9.5326e-04 | 0.9960 |

Least squares over the op's records that carry a token count (n); a alone is the mean when the token counts do not vary.
