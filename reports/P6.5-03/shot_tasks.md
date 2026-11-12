# P6.5-03: the shot's 30 tasks

Derived from the pulled logs of `P6.5-03` (`run_logs/agent/shot_plan.json`, `shot_attempts.jsonl`, `run.json`'s `arc26.evaluate` report, `submission.json` against `nvarc_submission.json`); DEV240, task ids and scores only. Record: [`docs/experiments/P6.5-03.md`](../../docs/experiments/P6.5-03.md).

Every task below had NVARC confidence 0 and NVARC score 0 in this run. Original: the DEV120 id the copy was made from (`settings.dev240.copies`). Slots: NVARC's $(N_1, N_2)$ per output, P a placeholder, R a real grid. Changed: the slots the merge rewrote (1 attempt 1, 2 attempt 2). `P6.3-05`: the original's score in that run's 15 (blank: not among them).

| Wave | Task | Original | Outputs | Tokens | Turn | Answered outputs | NVARC slots | Changed | Final score | `P6.3-05` |
|---|---|---|---|---:|---|---|---|---|---|---|
| 1 | `01863d60` | `5dbc8537` | 2 | 95,114 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 1 | `136b0064` | itself | 1 | 95,853 | cut, forced | 1 of 1 | PP | 12 | 0 | 1.0 |
| 1 | `14fd0455` | `446ef5d2` | 2 | 97,470 | cut, forced | 2 of 2 | PP, RP | 1, 2 | 0 | 0 |
| 1 | `1dd56dfa` | `80a900e0` | 1 | 95,195 | cut, forced | 1 of 1 | PP | 1 | 0 | 1.0 |
| 1 | `21897d95` | itself | 2 | 97,356 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 1 | `271d71e2` | itself | 1 | 95,637 | cut, forced | 0 of 1 | PP | - | 0 |  |
| 1 | `291dc1e1` | itself | 1 | 44,350 | ended | 1 of 1 | PP | 1 | **1** |  |
| 1 | `2db88f85` | `cbebaa4b` | 2 | 97,162 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 | 0 |
| 1 | `3b347aac` | `6ffbe589` | 1 | 88,977 | ended | 1 of 1 | PP | 12 | 0 |  |
| 1 | `3cbde633` | `a32d8b75` | 2 | 96,720 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 1 | `446ef5d2` | itself | 2 | 96,396 | cut, forced | 2 of 2 | PP, RP | 1, 2 | 0 | 0 |
| 1 | `4e34c42c` | itself | 2 | 48,988 | ended | 2 of 2 | PP, RP | 1, 2 | 0 |  |
| 1 | `511cb051` | `4e34c42c` | 2 | 79,338 | ended | 0 of 2 | PP, RP | -, - | 0 |  |
| 1 | `581f7754` | itself | 2 | 95,340 | cut, forced | 2 of 2 | PP, RR | 1, 2 | **0.5** | 0 |
| 1 | `5dbc8537` | itself | 2 | 95,144 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 2 | `65b59efc` | itself | 2 | 27,585 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 2 | `6e4f6532` | itself | 2 | 28,213 | cut, forced | 0 of 2 | PP, PP | -, - | 0 |  |
| 2 | `6ffbe589` | itself | 1 | 26,950 | cut, forced | 1 of 1 | PP | 1 | 0 |  |
| 2 | `77cdac68` | `21897d95` | 2 | 29,567 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 2 | `80a900e0` | itself | 1 | 27,445 | cut, forced | 0 of 1 | PP | - | 0 | 1.0 |
| 2 | `851445f9` | `291dc1e1` | 1 | 26,743 | cut, forced | 1 of 1 | PP | 1 | 0 |  |
| 2 | `86e8c6c8` | `65b59efc` | 2 | 28,869 | cut, forced | 2 of 2 | PP, PP | 1, 1 | 0 |  |
| 2 | `8da85b58` | `a47bf94d` | 1 | 27,892 | cut, forced | 1 of 1 | PP | 1 | 0 |  |
| 2 | `9012f7e7` | `6e4f6532` | 2 | 29,299 | cut, forced | 1 of 2 | PP, PP | 1, - | 0 |  |
| 2 | `9385bd28` | itself | 2 | 27,559 | cut, forced | 2 of 2 | PP, RR | 1, 2 | 0 | 0 |
| 2 | `9a3c5ce9` | `e12f9a14` | 2 | 28,287 | cut, forced | 1 of 2 | PP, PP | 1, - | 0 | 0 |
| 2 | `9cc9797f` | `136b0064` | 1 | 26,885 | cut, forced | 1 of 1 | PP | 1 | 0 | 1.0 |
| 2 | `a0319b81` | `f560132c` | 2 | 29,657 | cut, forced | 2 of 2 | PP, RP | 1, 2 | 0 | 0 |
| 2 | `a32d8b75` | itself | 2 | 29,607 | cut, forced | 0 of 2 | PP, PP | -, - | 0 |  |
| 2 | `a333fcf9` | `581f7754` | 2 | 27,076 | cut, forced | 0 of 2 | PP, RR | -, - | 0 | 0 |
