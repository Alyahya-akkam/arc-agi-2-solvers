# P5.2-01: version 11's development check (scored locally)

- **Decision: A** (Fault: no paired task).
- Boot to healthy 777.1 s; KV pool 243712 tokens; request limit 5; streams 5 (5 per arm); restarts 0; GPU peak MiB {'0': 21911.0, '1': 21911.0, '2': 21911.0, '3': 21911.0}.
- Paired tasks: 15; P6.9-01's 27B on them 6.0 (its cap 94,208; this run 49,152). A difference under 2 tasks is noise.

| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s | Forced share | Calls (failed) | Cached share | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tool | 8.5 | 8.5 of 15 | 747.8 | 31.342 | 0.267 | 95 (13) | 0.975 | 0 |

| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `97d7923e` | tool | 1.0 | 0.0 | stop | 17914 | 3 | 1 | 566.0 |
| `247ef758` | tool | 0.5 | 0.5 | stop | 29318 | 6 | 2 | 933.6 |
| `898e7135` | tool | 0.0 | 0.0 | length+force | 36380 | 3 | 0 | 1166.9 |
| `7b3084d4` | tool | 0.0 | 0.0 | length+force | 35907 | 12 | 0 | 1206.7 |
| `88e364bc` | tool | 0.5 | 0.0 | stop | 28692 | 11 | 2 | 937.5 |
| `142ca369` | tool | 0.0 | 0.0 | length+force | 36245 | 4 | 0 | 1548.6 |
| `a395ee82` | tool | 1.0 | 1.0 | stop | 13250 | 6 | 1 | 433.5 |
| `b5ca7ac4` | tool | 1.0 | 0.0 | stop | 17616 | 4 | 1 | 570.9 |
| `e8686506` | tool | 1.0 | 1.0 | stop | 17547 | 5 | 1 | 564.2 |
| `a25697e4` | tool | 0.0 | 0.0 | context+force | 37179 | 10 | 2 | 1209.9 |
| `b0039139` | tool | 0.5 | 1.0 | stop | 20956 | 8 | 1 | 677.6 |
| `1818057f` | tool | 0.0 | 1.0 | stop | 6865 | 3 | 0 | 215.3 |
| `7ed72f31` | tool | 1.0 | 0.5 | stop | 23531 | 7 | 2 | 747.8 |
| `6e453dd6` | tool | 1.0 | 1.0 | stop | 32547 | 6 | 1 | 938.3 |
| `7c66cb00` | tool | 1.0 | 0.0 | stop | 27783 | 7 | 1 | 723.6 |
