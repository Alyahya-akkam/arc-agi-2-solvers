# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 841.1 s; KV pool 340096 tokens; request limit 8; streams 7 (7 per arm); restarts 0; GPU peak MiB {'0': 22027.0, '1': 22027.0, '2': 22027.0, '3': 22027.0}.
- Paired tasks: 15; P6.9-01's 27B on them 6.0 (its cap 94,208; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 8.0 | 8.0 of 15 | 961.9 | 28.245 | 0.467 | 79 (10) | 0.965 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `247ef758` | tool | 0.5 | 0.5 | stop | 26205 | 4 | 2 | 961.9 |
| `898e7135` | tool | 0.0 | 0.0 | context+force | 34154 | 5 | 0 | 1253.9 |
| `7b3084d4` | tool | 0.0 | 0.0 | length+force | 35548 | 10 | 0 | 1319.6 |
| `a25697e4` | tool | 0.0 | 0.0 | length+force | 37380 | 5 | 0 | 1375.0 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 37955 | 2 | 1 | 1396.1 |
| `a395ee82` | tool | 1.0 | 1.0 | stop | 12540 | 3 | 0 | 465.2 |
| `97d7923e` | tool | 0.0 | 0.0 | length+force | 38830 | 5 | 0 | 1430.4 |
| `88e364bc` | tool | 0.0 | 0.0 | length+force | 43238 | 0 | 4 | 1586.6 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 7036 | 2 | 1 | 254.2 |
| `b5ca7ac4` | tool | 1.0 | 0.0 | stop | 16969 | 6 | 1 | 613.1 |
| `6e453dd6` | tool | 1.0 | 1.0 | stop | 15854 | 2 | 1 | 559.2 |
| `b0039139` | tool | 0.5 | 1.0 | stop | 22421 | 5 | 2 | 780.6 |
| `7ed72f31` | tool | 1.0 | 0.5 | stop | 21170 | 10 | 2 | 728.5 |
| `e8686506` | tool | 1.0 | 1.0 | stop | 30505 | 9 | 1 | 974.8 |
| `7c66cb00` | tool | 1.0 | 0.0 | length+force | 30451 | 11 | 0 | 895.9 |
