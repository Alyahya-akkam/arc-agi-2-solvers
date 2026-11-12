# P2.1-02 per task: gate verdicts against P2.1-01's, credit in three runs, the test prompts

Generated 3 Oct 2026 from the pulled logs by the experiment analyst of task P2.1-02; every column
is read from one file, as listed, with no other input. DEV40 (`arc26.data.DEV40`), 40 tasks in
id order; `PYTHONHASHSEED=0` in every run; P2.1-02 at `Code-commit` `84d0532`, P2.1-01 at
`b54ce14`, P1.3-01 at `74ed66d`.

| Column | Source and formula |
|---|---|
| $k$, gate (rank), candidates | the task's verdict line in `runs/P2.1-02/run_logs/r256-proj-emb-holdout-intest/gate/gate-rank*.jsonl` and, for P2.1-01, in `runs/P2.1-01/run_logs/r256-proj-emb-holdout/gate/` (`real_pairs`, `passed`, `expected_rank`: the held-out output's place in the gate's vote, none when no candidate equals it; `candidates`) |
| outputs | test outputs of the task, `docs/research/eval_task_stats.json` (`test_inputs`, K0) |
| gate s | the task's `gate` timing record, `runs/P2.1-02/run_logs/r256-proj-emb-holdout-intest/timing/` |
| $c_\text{in}$, $c_\text{on}$, $c_\text{off}$ | the task's credit in P2.1-02, P2.1-01, and P1.3-01 arm 1: the mean over its test outputs of membership in the `score_kgmon` correct set of each run's `selection.log` (arm 1's in `runs/P1.3-01/run_logs/r256-proj-emb/`, pulled in `~/wt-P1.3`), as `arc26.evaluate.TaskScore.score` |
| $T$, equal | training tokens, the `train` record's `tokens`; equal: whether P2.1-01's record holds the same count |
| batches, $\bar P$ | the task's `decode` records (test inputs only) and their mean prompt tokens, in P2.1-02 (in) and in P1.3-01 arm 1 (off) |

| Task | $k$ | Outputs | Gate (rank) | Gate in P2.1-01 | Candidates | Gate s | $c_\text{in}$ | $c_\text{on}$ | $c_\text{off}$ | $c_\text{in} - c_\text{on}$ | $c_\text{in} - c_\text{off}$ | $T$ | Equal to P2.1-01's | Batches in | Batches off | $\bar P_\text{in}$ | $\bar P_\text{off}$ |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `136b0064` | 3 | 1 | fail (none) | fail (none) | 5 | 60.6 | 0 | 0 | 0 | +0 | +0 | 69120 | yes | 4 | 4 | 4472 | 4472 |
| `16de56c4` | 3 | 2 | pass (1) | pass (1) | 20 | 76.7 | 0 | 0 | 0 | +0 | +0 | 81664 | yes | 8 | 8 | 4344 | 4344 |
| `195c6913` | 3 | 2 | fail (none) | fail (none) | 12 | 509.2 | 0 | 0 | 0 | +0 | +0 | 275456 | yes | 8 | 5 | 17576 | 17576 |
| `20a9e565` | 3 | 2 | fail (none) | fail (none) | 19 | 65.9 | 0 | 0 | 0 | +0 | +0 | 239424 | yes | 8 | 8 | 15094 | 15094 |
| `28a6681f` | 3 | 1 | fail (none) | fail (none) | 21 | 85.8 | 0 | 0 | 0 | +0 | +0 | 57856 | yes | 4 | 4 | 3176 | 3176 |
| `2ba387bc` | 4 | 1 | fail (2) | fail (2) | 26 | 161.3 | 1 | 1 | 1 | +0 | +0 | 234688 | yes | 4 | 4 | 11740 | 11740 |
| `31f7f899` | 3 | 1 | pass (1) | pass (1) | 24 | 43.0 | 0 | 1 | 0 | -1 | +0 | 109568 | yes | 4 | 4 | 5144 | 5144 |
| `332f06d7` | 4 | 1 | fail (2) | fail (2) | 22 | 88.1 | 0 | 0 | 0 | +0 | +0 | 165632 | yes | 4 | 4 | 7784 | 7784 |
| `3e6067c3` | 3 | 2 | fail (none) | fail (8) | 23 | 728.8 | 0 | 0 | 0 | +0 | +0 | 179840 | yes | 8 | 5 | 15316 | 15298 |
| `446ef5d2` | 2 | 2 | pass (1) | fail (4) | 16 | 95.4 | 0 | 0 | 0 | +0 | +0 | 66304 | yes | 8 | 8 | 5244 | 5244 |
| `45a5af55` | 2 | 1 | pass (1) | pass (1) | 20 | 346.7 | 1 | 1 | 1 | +0 | +0 | 117504 | yes | 4 | 4 | 7460 | 7460 |
| `4a21e3da` | 3 | 2 | fail (2) | pass (1) | 19 | 443.0 | 0 | 0 | 0 | +0 | +0 | 176640 | yes | 8 | 8 | 10068 | 10068 |
| `581f7754` | 3 | 2 | fail (none) | fail (none) | 10 | 166.5 | 0 | 0 | 0 | +0 | +0 | 99456 | yes | 8 | 8 | 6335 | 6335 |
| `5961cc34` | 4 | 1 | pass (1) | pass (1) | 20 | 314.4 | 0 | 0 | 0 | +0 | +0 | 344448 | yes | 4 | 4 | 18064 | 18064 |
| `64efde09` | 2 | 1 | fail (none) | fail (none) | 13 | 502.4 | 0 | 0 | 0 | +0 | +0 | 200832 | yes | 4 | 4 | 14532 | 14532 |
| `7b0280bc` | 3 | 1 | pass (1) | pass (1) | 18 | 300.7 | 1 | 1 | 1 | +0 | +0 | 150528 | yes | 4 | 4 | 11832 | 11832 |
| `7b5033c1` | 2 | 1 | fail (none) | fail (none) | 23 | 34.8 | 0 | 0 | 0 | +0 | +0 | 39872 | yes | 4 | 4 | 3616 | 3616 |
| `7c66cb00` | 3 | 1 | pass (1) | pass (1) | 19 | 581.2 | 1 | 0 | 1 | +1 | +0 | 384000 | yes | 4 | 4 | 21012 | 21012 |
| `80a900e0` | 2 | 1 | fail (none) | fail (none) | 15 | 730.7 | 0 | 0 | 0 | +0 | +0 | 76800 | yes | 4 | 4 | 10358 | 10358 |
| `8698868d` | 2 | 1 | fail (none) | fail (none) | 18 | 266.5 | 0 | 0 | 0 | +0 | +0 | 51776 | yes | 4 | 4 | 10194 | 10194 |
| `89565ca0` | 3 | 1 | pass (1) | pass (1) | 18 | 28.0 | 0 | 0 | 0 | +0 | +0 | 170752 | yes | 4 | 4 | 10906 | 10906 |
| `8f3a5a89` | 3 | 1 | fail (none) | fail (none) | 21 | 313.7 | 1 | 1 | 0 | +0 | +1 | 99328 | yes | 4 | 4 | 7136 | 7136 |
| `9385bd28` | 4 | 2 | fail (none) | fail (none) | 19 | 246.7 | 0 | 0 | 0.5 | +0 | -0.5 | 114560 | yes | 8 | 8 | 6972 | 6972 |
| `a25697e4` | 3 | 2 | fail (none) | fail (none) | 6 | 224.8 | 0 | 0 | 0 | +0 | +0 | 242944 | yes | 8 | 6 | 14827 | 14929 |
| `b0039139` | 4 | 2 | fail (2) | fail (none) | 18 | 14.4 | 0 | 0 | 0 | +0 | +0 | 68032 | yes | 8 | 8 | 3509 | 3509 |
| `b10624e5` | 2 | 1 | fail (none) | fail (none) | 6 | 465.0 | 0 | 0 | 0 | +0 | +0 | 194304 | yes | 4 | 4 | 15192 | 15192 |
| `b5ca7ac4` | 3 | 1 | pass (1) | pass (1) | 20 | 346.2 | 0 | 0 | 1 | +0 | -1 | 260608 | yes | 4 | 4 | 14264 | 14264 |
| `c7f57c3e` | 2 | 1 | fail (7) | fail (5) | 19 | 282.7 | 0 | 0 | 0 | +0 | +0 | 88320 | yes | 4 | 4 | 8792 | 8792 |
| `cbebaa4b` | 2 | 2 | fail (none) | fail (none) | 5 | 240.7 | 0 | 0 | 0 | +0 | +0 | 130304 | yes | 8 | 8 | 10976 | 10976 |
| `db0c5428` | 3 | 1 | fail (none) | fail (none) | 21 | 369.3 | 1 | 0 | 1 | +1 | +0 | 158720 | yes | 4 | 4 | 9272 | 9272 |
| `db695cfb` | 5 | 1 | pass (1) | pass (1) | 16 | 40.7 | 0 | 0 | 0 | +0 | +0 | 173312 | yes | 4 | 4 | 8376 | 8376 |
| `dbff022c` | 3 | 1 | fail (none) | fail (none) | 23 | 51.9 | 0 | 0 | 0 | +0 | +0 | 69632 | yes | 4 | 4 | 3508 | 3508 |
| `dd6b8c4b` | 3 | 2 | fail (none) | fail (none) | 16 | 61.7 | 0 | 0 | 0 | +0 | +0 | 69120 | yes | 8 | 8 | 3792 | 3792 |
| `de809cff` | 2 | 1 | fail (none) | fail (none) | 12 | 293.5 | 1 | 0 | 1 | +1 | +0 | 108288 | yes | 4 | 4 | 10512 | 10512 |
| `dfadab01` | 4 | 1 | fail (none) | fail (none) | 11 | 224.4 | 0 | 0 | 0 | +0 | +0 | 166144 | yes | 4 | 4 | 10280 | 10280 |
| `e12f9a14` | 4 | 2 | fail (2) | fail (2) | 16 | 119.0 | 0 | 0 | 0 | +0 | +0 | 281856 | yes | 8 | 8 | 13824 | 13824 |
| `e3721c99` | 2 | 2 | fail (none) | fail (none) | 11 | 859.0 | 0 | 0 | 0 | +0 | +0 | 238848 | yes | 8 | 5 | 18672 | 18672 |
| `e376de54` | 3 | 1 | fail (2) | fail (2) | 19 | 157.9 | 0 | 0 | 0 | +0 | +0 | 140800 | yes | 4 | 4 | 7712 | 7712 |
| `eee78d87` | 3 | 1 | fail (none) | fail (none) | 24 | 261.8 | 0 | 0 | 1 | +0 | -1 | 81920 | yes | 4 | 4 | 4032 | 4032 |
| `f560132c` | 2 | 2 | fail (none) | fail (none) | 3 | 55.3 | 0 | 0 | 0 | +0 | +0 | 63744 | yes | 8 | 8 | 7424 | 7424 |
