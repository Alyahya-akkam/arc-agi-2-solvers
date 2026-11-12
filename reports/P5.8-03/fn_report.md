# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 822.1 s; KV pool 260544 tokens; request limit 7; streams 6 (6 per arm); restarts 0; GPU peak MiB {'0': 21921.0, '1': 21921.0, '2': 21921.0, '3': 21921.0}.
- Paired tasks: 15; P5.2-01-r22 (version 18)'s 27B on them 7.0 (its cap 49,152; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 7.0 | 7.0 of 15 | 1042.4 | 29.616999999999997 | 0.733 | 74 (8) | 0.941 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `97d7923e` | tool | 1.0 | 1.0 | stop | 11973 | 5 | 1 | 407.9 |
| `88e364bc` | tool | 0.0 | 0.5 | length+force | 28487 | 10 | 2 | 987.0 |
| `247ef758` | tool | 1.0 | 0.0 | length+force | 30083 | 7 | 4 | 1042.4 |
| `7b3084d4` | tool | 0.0 | 0.0 | length+force | 32429 | 2 | 1 | 1125.5 |
| `898e7135` | tool | 0.0 | 0.0 | length+force | 33214 | 3 | 1 | 1153.0 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 33241 | 1 | 3 | 1153.8 |
| `a25697e4` | tool | 0.0 | 0.0 | length+force | 30222 | 5 | 2 | 1052.5 |
| `b5ca7ac4` | tool | 1.0 | 1.0 | stop | 15250 | 7 | 1 | 525.0 |
| `7ed72f31` | tool | 1.0 | 1.0 | stop | 17419 | 8 | 2 | 593.8 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 10521 | 6 | 1 | 360.8 |
| `a395ee82` | tool | 1.0 | 1.0 | length+force | 27636 | 5 | 1 | 949.5 |
| `e8686506` | tool | 0.0 | 0.0 | length+force | 33749 | 4 | 1 | 1112.2 |
| `b0039139` | tool | 0.0 | 0.5 | length+force | 35788 | 0 | 4 | 1173.8 |
| `7c66cb00` | tool | 1.0 | 0.0 | context+force | 23466 | 11 | 1 | 704.8 |
| `6e453dd6` | tool | 0.0 | 1.0 | length+force | 34971 | 0 | 1 | 1061.0 |
