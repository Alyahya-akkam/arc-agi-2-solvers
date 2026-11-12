# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 817.1 s; KV pool 170176 tokens; request limit 3; streams 3 (3 per arm); restarts 0; GPU peak MiB {'0': 21829.0, '1': 21829.0, '2': 21829.0, '3': 21829.0}.
- Paired tasks: 15; P6.9-01's 27B on them 6.0 (its cap 94,208; this run 65,536). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 9.5 | 9.5 of 15 | 1080.3 | 36.8295 | 0.467 | 93 (8) | 0.979 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `97d7923e` | tool | 0.0 | 0.0 | length+force | 47758 | 1 | 0 | 1248.2 |
| `247ef758` | tool | 0.5 | 0.5 | length+force | 48763 | 2 | 0 | 1276.3 |
| `7b3084d4` | tool | 1.0 | 0.0 | stop | 33669 | 14 | 1 | 1661.8 |
| `898e7135` | tool | 1.0 | 0.0 | length+force | 40944 | 10 | 1 | 1122.6 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 45008 | 5 | 2 | 1235.8 |
| `88e364bc` | tool | 0.5 | 0.0 | length+force | 41707 | 11 | 3 | 1146.7 |
| `a395ee82` | tool | 1.0 | 1.0 | stop | 13136 | 3 | 0 | 361.4 |
| `b5ca7ac4` | tool | 1.0 | 0.0 | stop | 16128 | 5 | 1 | 440.1 |
| `a25697e4` | tool | 0.0 | 0.0 | context+force | 46116 | 8 | 1 | 1262.6 |
| `b0039139` | tool | 0.5 | 1.0 | stop | 32867 | 5 | 2 | 896.7 |
| `e8686506` | tool | 1.0 | 1.0 | stop | 20199 | 6 | 1 | 554.3 |
| `7ed72f31` | tool | 1.0 | 0.5 | stop | 18243 | 7 | 1 | 501.5 |
| `6e453dd6` | tool | 1.0 | 1.0 | stop | 17788 | 3 | 1 | 489.4 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 24696 | 9 | 1 | 658.5 |
| `7c66cb00` | tool | 0.0 | 0.0 | length+force | 46148 | 4 | 2 | 1080.3 |
