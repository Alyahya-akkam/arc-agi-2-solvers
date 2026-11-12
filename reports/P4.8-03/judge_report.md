# P4.8-03: the accuracy of the pick-only judge's three arms on 41 outputs of P6.9-02 (DEV240)

**Decision:** Drop: every arm's change on the 41 outputs is at most 0 (A -10.50, B +0.00, judge -10.50).

**Engine:** 41 judgments with a reply, 0 with a degenerate reply; NaN lines in the engine log: 0 (engine_log_counts.json (or step.json's engine_log)); the judge's boot up: True.

Accuracy is hits over outputs with a pick (90% Wilson interval over outputs, then the clustered one over distinct original tasks, a DEV240 copy and its original one cluster); the change is in tasks of DEV240 against NVARC's own attempt 2. 1 task = 0.417 points.

| Arm | All | Rank 2 kept | Ranks 3 to 8 found |
|---|---|---|---|
| A, pick without thinking | 14/41 = 0.342 (0.23 to 0.47; 22 tasks: 0.20 to 0.52); overrides 29 (3 right); -10.50 tasks | 11/26 = 0.423 (0.28 to 0.58; 15 tasks: 0.24 to 0.63); overrides 15 (0 right); -11.83 tasks | 3/15 = 0.200 (0.08 to 0.41; 8 tasks: 0.06 to 0.49); overrides 14 (3 right); +1.33 tasks |
| B, pick with NVARC's prior (m = 0) | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |
| C, pick after a short look | 13/41 = 0.317 (0.21 to 0.44; 22 tasks: 0.18 to 0.49); overrides 24 (0 right); -10.50 tasks | 13/26 = 0.500 (0.35 to 0.65; 15 tasks: 0.30 to 0.70); overrides 13 (0 right); -10.50 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 11 (0 right); +0.00 tasks |
| rank2 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |
| A m=0.0 | 14/41 = 0.342 (0.23 to 0.47; 22 tasks: 0.20 to 0.52); overrides 29 (3 right); -10.50 tasks | 11/26 = 0.423 (0.28 to 0.58; 15 tasks: 0.24 to 0.63); overrides 15 (0 right); -11.83 tasks | 3/15 = 0.200 (0.08 to 0.41; 8 tasks: 0.06 to 0.49); overrides 14 (3 right); +1.33 tasks |
| A m=0.1 | 21/41 = 0.512 (0.39 to 0.64; 22 tasks: 0.35 to 0.68); overrides 15 (3 right); -4.50 tasks | 18/26 = 0.692 (0.53 to 0.82; 15 tasks: 0.48 to 0.85); overrides 8 (0 right); -5.83 tasks | 3/15 = 0.200 (0.08 to 0.41; 8 tasks: 0.06 to 0.49); overrides 7 (3 right); +1.33 tasks |
| A m=0.2 | 25/41 = 0.610 (0.48 to 0.72; 22 tasks: 0.44 to 0.76); overrides 3 (0 right); -0.33 tasks | 25/26 = 0.962 (0.84 to 0.99; 15 tasks: 0.79 to 0.99); overrides 1 (0 right); -0.33 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 2 (0 right); +0.00 tasks |
| A m=0.3 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 1 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 1 (0 right); +0.00 tasks |
| B m=0.0 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |
| B m=0.1 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |
| B m=0.2 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |
| B m=0.3 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |

The `m=` rows are a sweep of the override margin on the same 41 outputs: the best $m$ is chosen after seeing the answers, so those rows are optimistic and decide nothing.

| Arm | Band | Accuracy |
|---|---|---|
| A, pick without thinking | [0.5, 0.7) | 2/10 = 0.200 (0.07 to 0.46; 7 tasks: 0.06 to 0.51); overrides 8 (0 right); -2.00 tasks |
| A, pick without thinking | [0.7, 0.9) | 1/11 = 0.091 (0.02 to 0.32; 9 tasks: 0.02 to 0.35); overrides 9 (0 right); -4.83 tasks |
| A, pick without thinking | [0.9, 1] | 11/20 = 0.550 (0.37 to 0.72; 12 tasks: 0.33 to 0.75); overrides 12 (3 right); -3.67 tasks |
| B, pick with NVARC's prior (m = 0) | [0.5, 0.7) | 5/10 = 0.500 (0.27 to 0.73; 7 tasks: 0.24 to 0.76); overrides 0 (0 right); +0.00 tasks |
| B, pick with NVARC's prior (m = 0) | [0.7, 0.9) | 7/11 = 0.636 (0.39 to 0.82; 9 tasks: 0.37 to 0.84); overrides 0 (0 right); +0.00 tasks |
| B, pick with NVARC's prior (m = 0) | [0.9, 1] | 14/20 = 0.700 (0.52 to 0.84; 12 tasks: 0.46 to 0.86); overrides 0 (0 right); +0.00 tasks |
| C, pick after a short look | [0.5, 0.7) | 2/10 = 0.200 (0.07 to 0.46; 7 tasks: 0.06 to 0.51); overrides 7 (0 right); -2.00 tasks |
| C, pick after a short look | [0.7, 0.9) | 3/11 = 0.273 (0.11 to 0.52; 9 tasks: 0.10 to 0.55); overrides 7 (0 right); -3.50 tasks |
| C, pick after a short look | [0.9, 1] | 8/20 = 0.400 (0.24 to 0.58; 12 tasks: 0.21 to 0.63); overrides 10 (0 right); -5.00 tasks |
| rank2 | [0.5, 0.7) | 5/10 = 0.500 (0.27 to 0.73; 7 tasks: 0.24 to 0.76); overrides 0 (0 right); +0.00 tasks |
| rank2 | [0.7, 0.9) | 7/11 = 0.636 (0.39 to 0.82; 9 tasks: 0.37 to 0.84); overrides 0 (0 right); +0.00 tasks |
| rank2 | [0.9, 1] | 14/20 = 0.700 (0.52 to 0.84; 12 tasks: 0.46 to 0.86); overrides 0 (0 right); +0.00 tasks |

**Speed (arm C):** 41 judged in 459.5 s at 15 streams: 321.21 judgments per hour, 11.2 s per judgment; median thinking 2048.0 tokens, median judgment 145.352 s; 41 forced. Statuses: {'label': 34, 'unparsed': 7}.

**Ceiling and risk** (tasks, local rebuild / judge_ceiling.md): all [9.33, 20.33] / [9.33, 20.33]; [0.5, 0.7) [3.5, 3.0] / [3.5, 3.0]. Judgeable outputs (at least two pool grids): {'all': 248, '[0.7, 0.9)': 52, '[0.9, 1]': 109, '[0.5, 0.7)': 55, '(0, 0.5)': 26, '0': 6}.

## Variants of arm C (estimate, tasks of 240 against version 8)

Measured $q_\text{keep} = 0.5$, $q_\text{find} = 0.0$.

| Case | (1) rerank only | (2) judge, then the long shot | P4.7b alone |
|---|---|---|---|
| measured | -9.87 (judged 100%) | +0.00 (judged 100%) | +1.91 |
| q = 0.69 | +0.44 (judged 100%) | +2.99 (judged 100%) | +1.91 |
| q = 0.83 | +4.59 (judged 100%) | +3.90 (judged 100%) | +1.91 |
| q = 0.85 | +5.18 (judged 100%) | +4.03 (judged 100%) | +1.91 |
| q = 0.9 | +6.66 (judged 100%) | +4.35 (judged 100%) | +1.91 |

## Arms (each decided by `decide`, estimate in tasks of 240 against version 8)

| Arm | $q$ | $q_\text{keep}$ | $q_\text{find}$ | Change (41 outputs) | s per output | (1) rerank only | (2) judge, then the long shot | P4.7b alone | Decision |
|---|---|---|---|---|---|---|---|---|---|
| A, pick without thinking | 0.3415 | 0.4231 | 0.2 | -10.50 | 7.26 | -9.56 | +0.61 | +1.91 | Drop: the judge's change on the 41 outputs is -10.50 tasks (at most 0); (2) judge, then the long shot at +0.61 tasks of 240 against P4.7b alone at +1.91 |
| B, pick with NVARC's prior (m = 0) | 0.6341 | 1.0 | 0.0 | +0.00 | 7.3 | +0.30 | +1.64 | +1.91 | Drop: the judge's change on the 41 outputs is +0.00 tasks (at most 0); (2) judge, then the long shot at +1.64 tasks of 240 against P4.7b alone at +1.91 |
| C, pick after a short look | 0.3171 | 0.5 | 0.0 | -10.50 | 11.2 | -9.87 | +0.00 | +1.91 | Drop: the judge's change on the 41 outputs is -10.50 tasks (at most 0); (2) judge, then the long shot at +0.00 tasks of 240 against P4.7b alone at +1.91 |

**Arms A and B's speed:** 41 outputs, statuses {'ok': 41}, 328 passes in 597.0 s (14.56 s per output for both arms); prompt tokens A 689804, B 692920; the letters' median mass A 0.995868, B 0.997996 (the share of the first token's probability on the letters shown). Seconds per output per arm split the wall time by prompt tokens (estimate).

## Arm A's lead of rank 2

Right grid at rank 2: $\bar p_A(2) - \max_{r \ne 2} \bar p_A(r)$ per output, sorted (a margin $m$ keeps rank 2 on every output whose lead is above $-m$):

-0.201, -0.197, -0.195, -0.153, -0.150, -0.129, -0.103, -0.102, -0.087, -0.075, -0.035, -0.032, -0.027, -0.011, -0.001, +0.018, +0.024, +0.032, +0.044, +0.059, +0.059, +0.073, +0.096, +0.216, +0.431, +0.439

Right grid at ranks 3 to 8: $\bar p_A(r^*) - \bar p_A(2)$ per output, sorted (an override at margin $m$ needs a lead above $m$):

-0.119, -0.071, -0.070, -0.059, -0.041, -0.014, +0.006, +0.015, +0.033, +0.041, +0.044, +0.045, +0.108, +0.114, +0.124
