# P6.10-01: MTP (k = 1) against the base

| Measure | Value | |
|---|---|---|
| Row | Neutral |  |
| q (source) | 1.2135 (calls) |  |
| q from per-call medians | 1.2135 |  |

Reasons: a degenerate reply, NaN, or engine error in both steps

| Measure | shot_base | shot_mtp |
|---|---|---|
| Exit, end | 0, exited | 0, exited |
| Install ok | True | True |
| Boot s | 356.2 | 416.2 |
| KV pool (tokens) | 1746170 | 1507328 |
| Queued, shot, used | 15, 15, 12 | 15, 15, 13 |
| Caps | [16384] | [16384] |
| Full-batch R (tokens/s) | 342.916 | None |
| Per stream (tokens/s) | 22.861 | None |
| Window samples, seconds | 72, 710.86 | 1, 0.0 |
| Most running, waiting | 15.0, 13.0 | 15.0, 11.0 |
| Preemptions | 0.0 | 0.0 |
| Median per-call rate | 21.754 | 26.398 |
| Acceptance rate alpha | None | 0.8696 |
| Mean acceptance length L | None | 1.8696 |
| Degenerate (waves, calls) | 0, 0 | 0, 0 |
| Engine errors (waves, calls) | 0, 0 | 0, 0 |
| Engine log NaN, tracebacks | 0, 1 | 0, 1 |
| Engine gone | False | False |

KV for P6.9-02 at the full cap: MTP pool 1507328 on Modal, about 1467099 on Kaggle, against the need 1606845 (fits: False).
