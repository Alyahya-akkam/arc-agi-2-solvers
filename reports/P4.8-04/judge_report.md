# P4.8-04: the judge's accuracy on 41 outputs of P6.9-02 (DEV240)

**Decision:** No build: (2) judge, then the long shot at +0.08 tasks of 240 against P4.7b alone at +1.91, under the 1.0-task margin; P4.7b stays (the judge's change on the 37 finished outputs +0.83 tasks).

**Finished:** 37 of 41 judgments (26 ended their thinking on their own, 11 were forced at the cap); not finished: {'not started': 4}. Only finished judgments are scored, every row below on the same outputs: 23 of 26 right at rank 2, 14 of 15 right at ranks 3 to 8. Thinking tokens per finished judgment: median 54513.0, range 11978 to 65536.

**Engine:** 37 judgments with a reply, 0 with a degenerate reply; NaN lines in the engine log: 0 (engine_log_counts.json (or step.json's engine_log)); the judge's boot up: True.

Accuracy is hits over outputs with a pick (90% Wilson interval over outputs, then the clustered one over distinct original tasks, a DEV240 copy and its original one cluster); the change is in tasks of DEV240 against NVARC's own attempt 2. 1 task = 0.417 points.

| Arm | All | Rank 2 kept | Ranks 3 to 8 found |
|---|---|---|---|
| judge | 23/37 = 0.622 (0.49 to 0.74; 22 tasks: 0.45 to 0.77); overrides 15 (5 right); +0.83 tasks | 18/23 = 0.783 (0.62 to 0.89; 15 tasks: 0.57 to 0.91); overrides 5 (0 right); -2.50 tasks | 5/14 = 0.357 (0.19 to 0.57; 8 tasks: 0.15 to 0.64); overrides 10 (5 right); +3.33 tasks |
| raw | n/a | n/a | n/a |
| debiased | n/a | n/a | n/a |
| rank2 | 23/37 = 0.622 (0.49 to 0.74; 22 tasks: 0.45 to 0.77); overrides 0 (0 right); +0.00 tasks | 23/23 = 1.000 (0.89 to 1.00; 15 tasks: 0.85 to 1.00); overrides 0 (0 right); +0.00 tasks | 0/14 = 0.000 (0.00 to 0.16; 8 tasks: 0.00 to 0.25); overrides 0 (0 right); +0.00 tasks |

| Arm | Band | Accuracy |
|---|---|---|
| judge | [0.5, 0.7) | 3/10 = 0.300 (0.13 to 0.56; 7 tasks: 0.11 to 0.60); overrides 6 (1 right); -0.50 tasks |
| judge | [0.7, 0.9) | 9/10 = 0.900 (0.65 to 0.98; 9 tasks: 0.64 to 0.98); overrides 3 (2 right); +1.33 tasks |
| judge | [0.9, 1] | 11/17 = 0.647 (0.45 to 0.81; 12 tasks: 0.41 to 0.83); overrides 6 (2 right); +0.00 tasks |
| raw | [0.5, 0.7) | n/a |
| raw | [0.7, 0.9) | n/a |
| raw | [0.9, 1] | n/a |
| debiased | [0.5, 0.7) | n/a |
| debiased | [0.7, 0.9) | n/a |
| debiased | [0.9, 1] | n/a |
| rank2 | [0.5, 0.7) | 5/10 = 0.500 (0.27 to 0.73; 7 tasks: 0.24 to 0.76); overrides 0 (0 right); +0.00 tasks |
| rank2 | [0.7, 0.9) | 7/10 = 0.700 (0.44 to 0.87; 9 tasks: 0.43 to 0.88); overrides 0 (0 right); +0.00 tasks |
| rank2 | [0.9, 1] | 11/17 = 0.647 (0.45 to 0.81; 12 tasks: 0.41 to 0.83); overrides 0 (0 right); +0.00 tasks |

**Speed:** 37 judged in 8348.0 s at 15 streams: 15.96 judgments per hour, 225.6 s per judgment; median thinking 54513.0 tokens, median judgment 2964.213 s; 11 forced. Statuses: {'label': 37, 'not started': 4}.

**Ceiling and risk** (tasks, local rebuild / judge_ceiling.md): all [9.33, 20.33] / [9.33, 20.33]; [0.5, 0.7) [3.5, 3.0] / [3.5, 3.0]. Judgeable outputs (at least two pool grids): {'all': 248, '[0.7, 0.9)': 52, '[0.9, 1]': 109, '[0.5, 0.7)': 55, '(0, 0.5)': 26, '0': 6}.

## Variants (estimate, tasks of 240 against version 8)

Measured $q_\text{keep} = 0.7826$, $q_\text{find} = 0.3571$.

| Case | (1) rerank only | (2) judge, then the long shot | P4.7b alone |
|---|---|---|---|
| measured | -3.48 (judged 17%) | +0.08 (judged 24%) | +1.91 |
| q = 0.69 | -3.28 (judged 17%) | +0.29 (judged 24%) | +1.91 |
| q = 0.83 | -2.58 (judged 17%) | +0.51 (judged 24%) | +1.91 |
| q = 0.85 | -2.48 (judged 17%) | +0.54 (judged 24%) | +1.91 |
| q = 0.9 | -2.23 (judged 17%) | +0.62 (judged 24%) | +1.91 |
