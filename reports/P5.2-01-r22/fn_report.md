# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 832.1 s; KV pool 240256 tokens; request limit 6; streams 5 (5 per arm); restarts 0; GPU peak MiB {'0': 21911.0, '1': 21911.0, '2': 21911.0, '3': 21911.0}.
- Paired tasks: 15; P6.9-01's 27B on them 6.0 (its cap 94,208; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 7.0 | 7.0 of 15 | 1042.9 | 31.103 | 0.533 | 83 (12) | 0.977 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `97d7923e` | tool | 1.0 | 0.0 | stop | 21775 | 3 | 1 | 703.0 |
| `898e7135` | tool | 0.0 | 0.0 | length+force | 35779 | 4 | 1 | 1160.3 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 36091 | 2 | 2 | 1172.5 |
| `7b3084d4` | tool | 0.0 | 0.0 | context+force | 36571 | 15 | 1 | 1201.4 |
| `247ef758` | tool | 0.0 | 0.5 | length+force | 41326 | 1 | 3 | 1344.3 |
| `88e364bc` | tool | 0.5 | 0.0 | stop | 27965 | 12 | 2 | 914.5 |
| `a395ee82` | tool | 1.0 | 1.0 | stop | 19545 | 5 | 1 | 636.0 |
| `b5ca7ac4` | tool | 1.0 | 0.0 | stop | 22442 | 6 | 1 | 730.5 |
| `a25697e4` | tool | 0.0 | 0.0 | length+force | 35659 | 5 | 1 | 1161.1 |
| `7ed72f31` | tool | 1.0 | 0.5 | stop | 20852 | 6 | 1 | 676.9 |
| `1818057f` | tool | 1.0 | 1.0 | stop | 7157 | 3 | 0 | 235.9 |
| `b0039139` | tool | 0.5 | 1.0 | length+force | 40120 | 2 | 2 | 1299.2 |
| `e8686506` | tool | 0.0 | 1.0 | context+force | 40232 | 1 | 1 | 1265.9 |
| `6e453dd6` | tool | 1.0 | 1.0 | stop | 34140 | 11 | 1 | 1042.9 |
| `7c66cb00` | tool | 0.0 | 0.0 | length+force | 31696 | 7 | 2 | 825.6 |
