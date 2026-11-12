# P2.1-01 per task: gate verdicts, credit against P1.3-01 arm 1, and the held-out pair's size

Generated 2 Oct 2026 from the pulled logs by the experiment analyst of task P2.1-02; every column
is read from one file, as listed, with no other input. DEV40 (`arc26.data.DEV40`), 40 tasks in
id order; `PYTHONHASHSEED=0` in both runs; P2.1-01 at `Code-commit` `b54ce14`, P1.3-01 at
`74ed66d`.

| Column | Source and formula |
|---|---|
| $k$, held-out index, gate, candidates, rank, gate batches | the task's verdict line in `runs/P2.1-01/run_logs/r256-proj-emb-holdout/gate/gate-rank*.jsonl` (`real_pairs`, `held_out_index`, `passed`, `candidates`, `expected_rank`: the held-out output's place in the gate's vote, `none` when no candidate equals it); gate batches: its `gate_decode` records |
| outputs | test outputs of the task, `docs/research/eval_task_stats.json` (`test_inputs`, K0) |
| gate s | the task's `gate` timing record (the gate's span: its decode, rescoring, and vote), `runs/P2.1-01/run_logs/r256-proj-emb-holdout/timing/` |
| $c_\text{on}$, $c_\text{off}$ | the task's credit, the mean over its test outputs of membership in the `score_kgmon` correct set of `selection.log` (P2.1-01: `runs/P2.1-01/run_logs/r256-proj-emb-holdout/selection.log`; arm 1: `runs/P1.3-01/run_logs/r256-proj-emb/selection.log`, pulled in `~/wt-P1.3`), as `arc26.evaluate.TaskScore.score` |
| $T_\text{on}$, $T_\text{off}$ | training tokens, the `train` record's `tokens` (128 samples) in each run's timing folder |
| $h_\text{train}$ | $(T_\text{off} - T_\text{on}) / 128$: tokens of the held-out pair per training sample |
| $h_\text{dec}$ | $(\bar P_\text{off} - \bar P_\text{on}) / 4$, with $\bar P$ the mean `decode` record `tokens` per batch (4 prompts): tokens of the held-out pair per test prompt; n/a when the runs decoded a different number of batches (the 1200 s cap) |
| batches, $\bar P$ | the task's `decode` records (test inputs only; the gate's are `gate_decode`) and their mean prompt tokens |

| Task | $k$ | Held-out index | Outputs | Gate | Candidates | Rank | Gate s | Gate batches | $c_\text{on}$ | $c_\text{off}$ | Difference | $T_\text{on}$ | $T_\text{off}$ | $T_\text{on} / T_\text{off}$ | $(k - 1)/k$ | $h_\text{train}$ | $h_\text{dec}$ | Batches on | Batches off | $\bar P_\text{on}$ | $\bar P_\text{off}$ |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `136b0064` | 3 | 2 | 1 | fail | 1 | none | 53.6 | 4 | 0 | 0 | +0 | 69120 | 103680 | 0.667 | 0.667 | 270 | 270 | 4 | 4 | 3392 | 4472 |
| `16de56c4` | 3 | 2 | 2 | pass | 18 | 1 | 63.1 | 4 | 0 | 0 | +0 | 81664 | 112128 | 0.728 | 0.667 | 238 | 238 | 8 | 8 | 3392 | 4344 |
| `195c6913` | 3 | 2 | 2 | fail | 7 | none | 635.4 | 4 | 0 | 0 | +0 | 275456 | 442624 | 0.622 | 0.667 | 1306 | n/a | 7 | 5 | 12352 | 17576 |
| `20a9e565` | 3 | 2 | 2 | fail | 23 | none | 78.5 | 4 | 0 | 0 | +0 | 239424 | 367040 | 0.652 | 0.667 | 997 | 997 | 8 | 8 | 11106 | 15094 |
| `28a6681f` | 3 | 2 | 1 | fail | 21 | none | 96.1 | 4 | 0 | 0 | +0 | 57856 | 86784 | 0.667 | 0.667 | 226 | 226 | 4 | 4 | 2272 | 3176 |
| `2ba387bc` | 4 | 3 | 1 | fail | 27 | 2 | 164.3 | 4 | 1 | 1 | +0 | 234688 | 313792 | 0.748 | 0.750 | 618 | 618 | 4 | 4 | 9268 | 11740 |
| `31f7f899` | 3 | 2 | 1 | pass | 19 | 1 | 30.5 | 4 | 1 | 0 | +1 | 109568 | 124672 | 0.879 | 0.667 | 118 | 118 | 4 | 4 | 4672 | 5144 |
| `332f06d7` | 4 | 3 | 1 | fail | 22 | 2 | 86.6 | 4 | 0 | 0 | +0 | 165632 | 194560 | 0.851 | 0.750 | 226 | 226 | 4 | 4 | 6880 | 7784 |
| `3e6067c3` | 3 | 2 | 2 | fail | 17 | 8 | 533.7 | 4 | 0 | 0 | +0 | 179840 | 374144 | 0.481 | 0.667 | 1518 | n/a | 8 | 5 | 9244 | 15298 |
| `446ef5d2` | 2 | 1 | 2 | fail | 16 | 4 | 101.3 | 4 | 0 | 0 | +0 | 66304 | 95232 | 0.696 | 0.500 | 226 | 226 | 8 | 8 | 4340 | 5244 |
| `45a5af55` | 2 | 1 | 1 | pass | 18 | 1 | 339.4 | 4 | 1 | 1 | +0 | 117504 | 216640 | 0.542 | 0.500 | 774.5 | 774.5 | 4 | 4 | 4362 | 7460 |
| `4a21e3da` | 3 | 2 | 2 | pass | 19 | 1 | 458.3 | 4 | 0 | 0 | +0 | 176640 | 264960 | 0.667 | 0.667 | 690 | 690 | 8 | 8 | 7308 | 10068 |
| `581f7754` | 3 | 2 | 2 | fail | 11 | none | 197.6 | 4 | 0 | 0 | +0 | 99456 | 166400 | 0.598 | 0.667 | 523 | 523 | 8 | 8 | 4243 | 6335 |
| `5961cc34` | 4 | 3 | 1 | pass | 22 | 1 | 385.3 | 4 | 0 | 0 | +0 | 344448 | 480512 | 0.717 | 0.750 | 1063 | 1063 | 4 | 4 | 13812 | 18064 |
| `64efde09` | 2 | 1 | 1 | fail | 3 | none | 334.2 | 4 | 0 | 0 | +0 | 200832 | 352896 | 0.569 | 0.500 | 1188 | 1188 | 4 | 4 | 9780 | 14532 |
| `7b0280bc` | 3 | 2 | 1 | pass | 21 | 1 | 371.4 | 4 | 1 | 1 | +0 | 150528 | 258816 | 0.582 | 0.667 | 846 | 846 | 4 | 4 | 8448 | 11832 |
| `7b5033c1` | 2 | 1 | 1 | fail | 26 | none | 45.2 | 4 | 0 | 0 | +0 | 39872 | 80128 | 0.498 | 0.500 | 314.5 | 314.5 | 4 | 4 | 2358 | 3616 |
| `7c66cb00` | 3 | 2 | 1 | pass | 21 | 1 | 746.3 | 4 | 0 | 1 | -1 | 384000 | 576000 | 0.667 | 0.667 | 1500 | 1500 | 4 | 4 | 15012 | 21012 |
| `80a900e0` | 2 | 1 | 1 | fail | 13 | none | 598.9 | 4 | 0 | 0 | +0 | 76800 | 268800 | 0.286 | 0.500 | 1500 | 1500 | 4 | 4 | 4358 | 10358 |
| `8698868d` | 2 | 1 | 1 | fail | 21 | none | 266.1 | 4 | 0 | 0 | +0 | 51776 | 206400 | 0.251 | 0.500 | 1208 | 1208 | 4 | 4 | 5362 | 10194 |
| `89565ca0` | 3 | 2 | 1 | pass | 21 | 1 | 29.0 | 4 | 0 | 0 | +0 | 170752 | 233088 | 0.733 | 0.667 | 487 | 487 | 4 | 4 | 8958 | 10906 |
| `8f3a5a89` | 3 | 2 | 1 | fail | 20 | none | 298.5 | 4 | 1 | 0 | +1 | 99328 | 207616 | 0.478 | 0.667 | 846 | 846 | 4 | 4 | 3752 | 7136 |
| `9385bd28` | 4 | 3 | 2 | fail | 18 | none | 229.5 | 4 | 0 | 0.5 | -0.5 | 114560 | 188672 | 0.607 | 0.750 | 579 | 579 | 8 | 8 | 4656 | 6972 |
| `a25697e4` | 3 | 2 | 2 | fail | 13 | none | 267.5 | 4 | 0 | 0 | +0 | 242944 | 364416 | 0.667 | 0.667 | 949 | n/a | 8 | 6 | 11031 | 14929 |
| `b0039139` | 4 | 3 | 2 | fail | 16 | none | 13.2 | 4 | 0 | 0 | +0 | 68032 | 84672 | 0.803 | 0.750 | 130 | 130 | 8 | 8 | 2989 | 3509 |
| `b10624e5` | 2 | 1 | 1 | fail | 5 | none | 481.6 | 4 | 0 | 0 | +0 | 194304 | 388608 | 0.500 | 0.500 | 1518 | 1518 | 4 | 4 | 9120 | 15192 |
| `b5ca7ac4` | 3 | 2 | 1 | pass | 22 | 1 | 389.8 | 4 | 0 | 1 | -1 | 260608 | 390912 | 0.667 | 0.667 | 1018 | 1018 | 4 | 4 | 10192 | 14264 |
| `c7f57c3e` | 2 | 1 | 1 | fail | 18 | 5 | 354.5 | 4 | 0 | 0 | +0 | 88320 | 176640 | 0.500 | 0.500 | 690 | 690 | 4 | 4 | 6032 | 8792 |
| `cbebaa4b` | 2 | 1 | 2 | fail | 5 | none | 228.6 | 4 | 0 | 0 | +0 | 130304 | 260608 | 0.500 | 0.500 | 1018 | 1018 | 8 | 8 | 6904 | 10976 |
| `db0c5428` | 3 | 2 | 1 | fail | 20 | none | 383.2 | 4 | 0 | 1 | -1 | 158720 | 256768 | 0.618 | 0.667 | 766 | 766 | 4 | 4 | 6208 | 9272 |
| `db695cfb` | 5 | 4 | 1 | pass | 16 | 1 | 40.4 | 4 | 0 | 0 | +0 | 173312 | 197120 | 0.879 | 0.800 | 186 | 186 | 4 | 4 | 7632 | 8376 |
| `dbff022c` | 3 | 2 | 1 | fail | 21 | none | 51.4 | 4 | 0 | 0 | +0 | 69632 | 88192 | 0.790 | 0.667 | 145 | 145 | 4 | 4 | 2928 | 3508 |
| `dd6b8c4b` | 3 | 2 | 2 | fail | 18 | none | 73.4 | 4 | 0 | 0 | +0 | 69120 | 103680 | 0.667 | 0.667 | 270 | 270 | 8 | 8 | 2712 | 3792 |
| `de809cff` | 2 | 1 | 1 | fail | 17 | none | 289.0 | 4 | 0 | 1 | -1 | 108288 | 216576 | 0.500 | 0.500 | 846 | 846 | 4 | 4 | 7128 | 10512 |
| `dfadab01` | 4 | 3 | 1 | fail | 11 | none | 255.1 | 4 | 0 | 0 | +0 | 166144 | 274432 | 0.605 | 0.750 | 846 | 846 | 4 | 4 | 6896 | 10280 |
| `e12f9a14` | 4 | 3 | 2 | fail | 17 | 2 | 121.1 | 4 | 0 | 0 | +0 | 281856 | 322560 | 0.874 | 0.750 | 318 | 318 | 8 | 8 | 12552 | 13824 |
| `e3721c99` | 2 | 1 | 2 | fail | 10 | none | 861.5 | 4 | 0 | 0 | +0 | 238848 | 477696 | 0.500 | 0.500 | 1866 | n/a | 7 | 5 | 11208 | 18672 |
| `e376de54` | 3 | 2 | 1 | fail | 19 | 2 | 223.2 | 4 | 0 | 0 | +0 | 140800 | 211200 | 0.667 | 0.667 | 550 | 550 | 4 | 4 | 5512 | 7712 |
| `eee78d87` | 3 | 2 | 1 | fail | 29 | none | 243.7 | 4 | 0 | 1 | -1 | 81920 | 122880 | 0.667 | 0.667 | 320 | 320 | 4 | 4 | 2752 | 4032 |
| `f560132c` | 2 | 1 | 2 | fail | 4 | none | 52.4 | 4 | 0 | 0 | +0 | 63744 | 132352 | 0.482 | 0.500 | 536 | 536 | 8 | 8 | 5280 | 7424 |
