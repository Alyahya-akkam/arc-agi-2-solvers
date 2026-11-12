# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 766.1 s; KV pool 377856 tokens; request limit 9; streams 8 (8 per arm); restarts 2; GPU peak MiB {'0': 22227.0, '1': 22227.0, '2': 22227.0, '3': 22227.0}.
- Paired tasks: 15; P6.9-01's 27B on them 6.0 (its cap 94,208; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 4.5 | 4.5 of 15 | 1081.8 | 26.563 | 0.6 | 82 (14) | 0.966 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `a25697e4` | tool | 0.0 | 0.0 | abort | 0 | 0 | 0 | 19.0 |
| `a395ee82` | tool | 0.0 | 1.0 | abort | 0 | 0 | 0 | 16.0 |
| `b5ca7ac4` | tool | 1.0 | 0.0 | stop | 20708 | 5 | 1 | 788.6 |
| `247ef758` | tool | 0.5 | 0.5 | stop | 25076 | 5 | 2 | 968.2 |
| `88e364bc` | tool | 0.5 | 0.0 | length+force | 29709 | 18 | 2 | 1143.7 |
| `7b3084d4` | tool | 0.0 | 0.0 | length+force | 34535 | 10 | 2 | 1325.0 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 36742 | 2 | 0 | 1419.7 |
| `97d7923e` | tool | 0.0 | 0.0 | length+force | 37063 | 3 | 0 | 1422.0 |
| `898e7135` | tool | 0.0 | 0.0 | context+force | 37292 | 5 | 1 | 1432.0 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 8718 | 3 | 1 | 322.4 |
| `b0039139` | tool | 0.0 | 1.0 | length+force | 43375 | 2 | 0 | 1646.9 |
| `6e453dd6` | tool | 1.0 | 1.0 | stop | 24366 | 4 | 1 | 841.4 |
| `7ed72f31` | tool | 0.5 | 0.5 | context+force | 30972 | 15 | 1 | 1081.8 |
| `e8686506` | tool | 0.0 | 1.0 | length+force | 37190 | 2 | 2 | 1299.7 |
| `7c66cb00` | tool | 0.0 | 0.0 | context+force | 33320 | 8 | 2 | 960.9 |
