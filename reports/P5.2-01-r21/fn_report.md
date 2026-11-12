# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 792.1 s; KV pool 240256 tokens; request limit 6; streams 5 (5 per arm); restarts 0; GPU peak MiB {'0': 21911.0, '1': 21911.0, '2': 21911.0, '3': 21911.0}.
- Paired tasks: 15; P6.9-01's 27B on them 6.0 (its cap 94,208; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 8.0 | 8.0 of 15 | 1024.1 | 31.3795 | 0.467 | 86 (7) | 0.972 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `97d7923e` | tool | 1.0 | 0.0 | stop | 31248 | 9 | 1 | 1024.1 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 36325 | 4 | 4 | 1190.7 |
| `247ef758` | tool | 0.5 | 0.5 | context+force | 36608 | 7 | 2 | 1202.0 |
| `7b3084d4` | tool | 0.0 | 0.0 | length+force | 38201 | 5 | 2 | 1254.2 |
| `898e7135` | tool | 0.0 | 0.0 | length+force | 38929 | 6 | 0 | 1278.5 |
| `a395ee82` | tool | 1.0 | 1.0 | stop | 25670 | 6 | 1 | 821.9 |
| `b0039139` | tool | 0.0 | 1.0 | stop | 25684 | 4 | 2 | 822.1 |
| `b5ca7ac4` | tool | 1.0 | 0.0 | stop | 27830 | 6 | 1 | 895.7 |
| `88e364bc` | tool | 0.5 | 0.0 | length+force | 36947 | 8 | 2 | 1200.1 |
| `a25697e4` | tool | 0.0 | 0.0 | length+force | 38085 | 2 | 1 | 1228.3 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 6939 | 6 | 1 | 226.9 |
| `e8686506` | tool | 1.0 | 1.0 | stop | 21917 | 7 | 1 | 697.2 |
| `6e453dd6` | tool | 1.0 | 1.0 | stop | 23209 | 4 | 1 | 712.2 |
| `7c66cb00` | tool | 1.0 | 0.0 | stop | 20415 | 10 | 1 | 586.3 |
| `7ed72f31` | tool | 0.0 | 0.5 | length+force | 37322 | 2 | 0 | 1066.7 |
