# MT-01 encode timing

Records: 813; skipped lines: 0; files: `runs/MT-01/run_logs/timing-rank0.jsonl`, `runs/MT-01/run_logs/timing-rank1.jsonl`, `runs/MT-01/run_logs/timing-rank2.jsonl`, `runs/MT-01/run_logs/timing-rank3.jsonl`.

## Per puzzle

No puzzle records.

## Ops outside puzzles

| Op | Records | Total s | Mean s |
|---|---:|---:|---:|
| encode | 813 | 106298.74 | 130.75 |

## Cost model: seconds = a + b * tokens

| Op | n | a (s) | b (s per token) | R^2 |
|---|---:|---:|---:|---:|
| encode | 813 | -0.7419 | 1.6656e-04 | 0.9660 |

Least squares over the op's records that carry a token count (n); a alone is the mean when the token counts do not vary.
