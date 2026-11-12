# P4.8-02: the judge's accuracy on 41 outputs of P6.9-02 (DEV240)

**Decision:** Drop: the judge's change on the 41 outputs is -10.17 tasks (at most 0); (2) judge, then the long shot at -0.92 tasks of 240 against P4.7b alone at +1.91.

**Engine:** 41 judgments with a reply, 0 with a degenerate reply; NaN lines in the engine log: 0 (engine_log_counts.json (or step.json's engine_log)); the judge's boot up: True.

Accuracy is hits over outputs with a pick (90% Wilson interval over outputs, then the clustered one over distinct original tasks, a DEV240 copy and its original one cluster); the change is in tasks of DEV240 against NVARC's own attempt 2. 1 task = 0.417 points.

| Arm | All | Rank 2 kept | Ranks 3 to 8 found |
|---|---|---|---|
| judge | 12/41 = 0.293 (0.19 to 0.42; 22 tasks: 0.16 to 0.47); overrides 31 (5 right); -10.17 tasks | 7/26 = 0.269 (0.15 to 0.43; 15 tasks: 0.13 to 0.48); overrides 19 (0 right); -13.33 tasks | 5/15 = 0.333 (0.17 to 0.54; 8 tasks: 0.13 to 0.62); overrides 12 (5 right); +3.17 tasks |
| raw | 8/41 = 0.195 (0.11 to 0.31; 22 tasks: 0.09 to 0.36); overrides 31 (2 right); -14.33 tasks | 6/26 = 0.231 (0.12 to 0.39; 15 tasks: 0.10 to 0.44); overrides 20 (0 right); -15.33 tasks | 2/15 = 0.133 (0.05 to 0.33; 8 tasks: 0.03 to 0.42); overrides 11 (2 right); +1.00 tasks |
| debiased | 8/41 = 0.195 (0.11 to 0.31; 22 tasks: 0.09 to 0.36); overrides 32 (2 right); -13.00 tasks | 6/26 = 0.231 (0.12 to 0.39; 15 tasks: 0.10 to 0.44); overrides 20 (0 right); -14.33 tasks | 2/15 = 0.133 (0.05 to 0.33; 8 tasks: 0.03 to 0.42); overrides 12 (2 right); +1.33 tasks |
| rank2 | 26/41 = 0.634 (0.51 to 0.75; 22 tasks: 0.46 to 0.78); overrides 0 (0 right); +0.00 tasks | 26/26 = 1.000 (0.91 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/15 = 0.000 (0.00 to 0.15; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |

| Arm | Band | Accuracy |
|---|---|---|
| judge | [0.5, 0.7) | 2/10 = 0.200 (0.07 to 0.46; 7 tasks: 0.06 to 0.51); overrides 9 (2 right); -1.00 tasks |
| judge | [0.7, 0.9) | 4/11 = 0.364 (0.18 to 0.61; 9 tasks: 0.16 to 0.63); overrides 8 (2 right); -3.17 tasks |
| judge | [0.9, 1] | 6/20 = 0.300 (0.16 to 0.48; 12 tasks: 0.14 to 0.54); overrides 14 (1 right); -6.00 tasks |
| raw | [0.5, 0.7) | 2/10 = 0.200 (0.07 to 0.46; 7 tasks: 0.06 to 0.51); overrides 10 (2 right); -2.00 tasks |
| raw | [0.7, 0.9) | 1/11 = 0.091 (0.02 to 0.32; 9 tasks: 0.02 to 0.35); overrides 10 (0 right); -4.83 tasks |
| raw | [0.9, 1] | 5/20 = 0.250 (0.13 to 0.43; 12 tasks: 0.10 to 0.49); overrides 11 (0 right); -7.50 tasks |
| debiased | [0.5, 0.7) | 1/10 = 0.100 (0.02 to 0.35; 7 tasks: 0.02 to 0.41); overrides 10 (1 right); -2.00 tasks |
| debiased | [0.7, 0.9) | 3/11 = 0.273 (0.11 to 0.52; 9 tasks: 0.10 to 0.55); overrides 8 (1 right); -3.50 tasks |
| debiased | [0.9, 1] | 4/20 = 0.200 (0.09 to 0.38; 12 tasks: 0.07 to 0.44); overrides 14 (0 right); -7.50 tasks |
| rank2 | [0.5, 0.7) | 5/10 = 0.500 (0.27 to 0.73; 7 tasks: 0.24 to 0.76); overrides 0 (0 right); +0.00 tasks |
| rank2 | [0.7, 0.9) | 7/11 = 0.636 (0.39 to 0.82; 9 tasks: 0.37 to 0.84); overrides 0 (0 right); +0.00 tasks |
| rank2 | [0.9, 1] | 14/20 = 0.700 (0.52 to 0.84; 12 tasks: 0.46 to 0.86); overrides 0 (0 right); +0.00 tasks |

**Speed:** 41 judged in 2050.4 s at 15 streams: 71.98 judgments per hour, 50.0 s per judgment; median thinking 12288.0 tokens, median judgment 697.908 s; 41 forced. Statuses: {'label': 39, 'unsure': 2}.

**Ceiling and risk** (tasks, local rebuild / judge_ceiling.md): all [9.33, 20.33] / [9.33, 20.33]; [0.5, 0.7) [3.5, 3.0] / [3.5, 3.0]. Judgeable outputs (at least two pool grids): {'all': 248, '[0.7, 0.9)': 52, '[0.9, 1]': 109, '[0.5, 0.7)': 55, '(0, 0.5)': 26, '0': 6}.

## Variants (estimate, tasks of 240 against version 8)

Measured $q_\text{keep} = 0.2692$, $q_\text{find} = 0.3333$.

| Case | (1) rerank only | (2) judge, then the long shot | P4.7b alone |
|---|---|---|---|
| measured | -12.18 (judged 76%) | -0.92 (judged 100%) | +1.91 |
| q = 0.69 | -3.20 (judged 76%) | +1.59 (judged 100%) | +1.91 |
| q = 0.83 | -0.06 (judged 76%) | +2.50 (judged 100%) | +1.91 |
| q = 0.85 | +0.39 (judged 76%) | +2.63 (judged 100%) | +1.91 |
| q = 0.9 | +1.51 (judged 76%) | +2.95 (judged 100%) | +1.91 |
