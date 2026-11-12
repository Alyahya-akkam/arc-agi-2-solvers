# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 877.1 s; KV pool 271552 tokens; request limit 7; streams 6 (6 per arm); restarts 0; GPU peak MiB {'0': 21961.0, '1': 21961.0, '2': 21961.0, '3': 21961.0}.
- Paired tasks: 15; P5.2-01-r22 (version 18)'s 27B on them 7.0 (its cap 49,152; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 6.5 | 6.5 of 15 | 914.3 | 29.665 | 0.533 | 90 (9) | 0.968 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `97d7923e` | tool | 1.0 | 1.0 | stop | 18332 | 4 | 1 | 624.2 |
| `88e364bc` | tool | 0.5 | 0.5 | length+force | 28491 | 13 | 0 | 974.7 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 28899 | 5 | 0 | 989.9 |
| `898e7135` | tool | 0.0 | 0.0 | context+force | 34355 | 10 | 1 | 1178.8 |
| `7b3084d4` | tool | 0.0 | 0.0 | length+force | 37022 | 3 | 2 | 1270.1 |
| `247ef758` | tool | 0.0 | 0.0 | length+force | 39503 | 2 | 4 | 1353.4 |
| `a395ee82` | tool | 1.0 | 1.0 | stop | 14582 | 4 | 0 | 502.5 |
| `a25697e4` | tool | 0.0 | 0.0 | length+force | 26520 | 7 | 0 | 914.3 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 8235 | 3 | 1 | 282.7 |
| `e8686506` | tool | 1.0 | 0.0 | stop | 18191 | 7 | 1 | 626.3 |
| `6e453dd6` | tool | 0.0 | 1.0 | stop | 13749 | 2 | 1 | 472.9 |
| `b5ca7ac4` | tool | 0.0 | 1.0 | context+force | 28125 | 8 | 0 | 965.9 |
| `7ed72f31` | tool | 1.0 | 1.0 | stop | 22234 | 7 | 2 | 736.9 |
| `b0039139` | tool | 1.0 | 0.5 | stop | 28511 | 6 | 2 | 941.2 |
| `7c66cb00` | tool | 0.0 | 0.0 | length+force | 31735 | 9 | 2 | 800.2 |
