# P6.3-05: merging the shot's answers with NVARC's attempts

Input: `P6.3-05`'s attempts (`runs/P6.3-05/run_logs/direct_attempts.jsonl`, scored on Kaggle in
`shot_metrics.json`), `P2.6-01`'s attempts (`submission.json`) and per-candidate rows
(`reports/P2.8/P2.6-01.csv`), and for the buckets also `P1.1-01`'s (`reports/P2.8/P1.1-01.csv`,
its `submission.json`). Code-commit of the shot `acac69d`, seed 0. Experiment record:
[`docs/experiments/P6.3-05.md`](../../docs/experiments/P6.3-05.md).

**Method.** No solutions file was read. An output's right grid is known only where a log already
scored it: an NVARC attempt row with `correct` 1, or the shot's answer on a task its
`arc26.evaluate` score puts at 1 (every output right) or 0 (every output wrong); the model gave one
answer per output, so a right output is right by $S_1$. Every other output's attempts are all
scored wrong by the logs, so a sentinel grid unequal to every attempt stands in for its solution.
Each merge rule's submission is then scored by `arc26.evaluate.score_submission`; the shot's own
pair $(S_1, S_2)$ reproduces its logged 4.5, and the buckets reproduce `P2.6-01`'s 38.33 and
`P1.1-01`'s 37.17 of 120 (DEV120). One gap: `3e6067c3` scores 0.5, so one of its two outputs is
right by $S_1$, and the logs do not say which. Both assignments give identical numbers for every
rule below, because NVARC is wrong on both outputs and every rule that holds $S_1$ holds it on both.

## Per output (26 test outputs, 15 tasks)

$N_1, N_2$: NVARC's attempts in `P2.6-01` ("real" when the CSV has a candidate in that slot, else
`[[0]]`); $S_1, S_2$: the shot's first two different answers. `merge.csv` has the same rows with
NVARC's top `softmax_p` per output and the planned rule's pair.

| Task, output | $N_1$ real | $N_1$ right | $N_2$ right | Shot answers | $S_1$ right | $S_2$ right | $S_1 = N_1$ | Planned pair |
|---|---|---|---|---|---|---|---|---|
| `136b0064` 0 | no | no | no | 1 | **yes** | n/a | no | $(S_1, N_2)$ |
| `195c6913` 0 | no | no | no | 0 | no | n/a | no | $(N_1, N_2)$ |
| `195c6913` 1 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `446ef5d2` 0 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `446ef5d2` 1 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `581f7754` 0 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `581f7754` 1 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `80a900e0` 0 | no | no | no | 1 | **yes** | n/a | no | $(S_1, N_2)$ |
| `9385bd28` 0 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `9385bd28` 1 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `cbebaa4b` 0 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `cbebaa4b` 1 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `e12f9a14` 0 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `e12f9a14` 1 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `e3721c99` 0 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `e3721c99` 1 | no | no | no | 0 | no | n/a | no | $(N_1, N_2)$ |
| `f560132c` 0 | no | no | no | 1 | no | n/a | no | $(S_1, N_2)$ |
| `f560132c` 1 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `16de56c4` 0 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `16de56c4` 1 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `7b5033c1` 0 | yes | no | no | 1 | **yes** | n/a | no | $(N_1, S_1)$ |
| `3e6067c3` 0 | yes | no | no | 1 | one of the two | n/a | no | $(N_1, S_1)$ |
| `3e6067c3` 1 | yes | no | no | 1 | one of the two | n/a | no | $(N_1, S_1)$ |
| `28a6681f` 0 | yes | no | no | 1 | no | n/a | no | $(N_1, S_1)$ |
| `dd6b8c4b` 0 | yes | no | no | 1 | **yes** | n/a | no | $(N_1, S_1)$ |
| `dd6b8c4b` 1 | yes | no | no | 1 | **yes** | n/a | no | $(N_1, S_1)$ |

Totals: $N_1$ or $N_2$ right on 0 of 26; $S_1$ right on 6 of 26; no $S_2$ on any output (the
model wrote one answer per test input, so $S_2$ is the `[[0]]` pad); $S_1 = N_1$ on 0 of 26.
Planned pairs: $(N_1, S_1)$ on 14 outputs, $(S_1, N_2)$ with $N_2$ = `[[0]]` (no NVARC candidate,
no $S_2$) on 10, NVARC's attempts on 2 (no shot answer, and NVARC had none either).

## Merge rules on the 15 tasks

| Rule | Score (`arc26.evaluate`) | Tasks solved | Outputs right |
|---|---|---|---|
| NVARC alone $(N_1, N_2)$ | 0.0 of 15 | 0 of 15 | 0 of 26 |
| Planned: $(N_1, S_1)$ when $N_1$ is real and $S_1 \ne N_1$; $(N_1, N_2)$ when $S_1 = N_1$; $(S_1, S_2)$ or $(S_1, N_2)$ without an NVARC candidate; $(N_1, N_2)$ without a shot answer | **4.5 of 15** | 4 of 15 | 6 of 26 |
| $(S_1, S_2)$ (the shot alone) | 4.5 of 15 | 4 of 15 | 6 of 26 |
| $(S_1, N_1)$ | 4.5 of 15 | 4 of 15 | 6 of 26 |

The three rules that hold $S_1$ tie: on these 15 NVARC is right nowhere and $S_2$ never exists,
so the rules differ only in which wrong grid fills the second slot. The tie says nothing about
which is better; the choice rests on the buckets below. The difference against NVARC alone,
4 tasks, is past the 2-task DEV40 noise floor.

## NVARC by task confidence (DEV120, 120 tasks each)

Task confidence: the minimum over a task's outputs of the `rank_kgmon` 1 row's `softmax_p`, 0 for
an output without candidates. Solved: the task scores 1 (`arc26.evaluate`). "Only attempt 2": the
output is right by $N_2$ and not by $N_1$, the outputs a rule that replaces $N_2$ with $S_1$
would lose.

| Bucket | `P2.6-01` tasks | solved | score | outputs | right | only attempt 2 | `P1.1-01` tasks | solved | score | outputs | right | only attempt 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $[0, 0.1)$ | 23 | 0 (0%) | 0.0 | 40 | 0 | 0 | 10 | 0 (0%) | 0.0 | 19 | 0 | 0 |
| $[0.1, 0.3)$ | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| $[0.3, 0.5)$ | 11 | 0 (0%) | 0.0 | 16 | 0 | 0 | 9 | 1 (11%) | 1.0 | 12 | 1 | 0 |
| $[0.5, 0.7)$ | 23 | 6 (26%) | 7.67 | 37 | 13 | **3 (8.1%)** | 23 | 4 (17%) | 5.83 | 38 | 9 | **5 (13.2%)** |
| $[0.7, 1]$ | 63 | 29 (46%) | 30.67 | 79 | 38 | 9 (11.4%) | 78 | 27 (35%) | 30.33 | 103 | 41 | 6 (5.8%) |
| All | 120 | 35 | 38.33 | 172 | 51 | 12 | 120 | 32 | 37.17 | 172 | 51 | 11 |

- Every task in $[0, 0.1)$ has confidence exactly 0 (an output without candidates); no task falls in
  $[0.1, 0.3)$ (the lowest nonzero confidence is 0.381 in `P2.6-01`, 0.342 in `P1.1-01`).
- **Below 0.5** NVARC solves 0 of 34 tasks (`P2.6-01`) and 1 of 19 (`P1.1-01`), and no output
  there is right by attempt 2 alone (0 of 56, 0 of 31); the one right output in `P1.1-01` is
  attempt 1, which the planned rule keeps.
- **In $[0.5, 0.7)$** NVARC solves 6 of 23 and 4 of 23, and 3 of 37 and 5 of 38 outputs are right
  by attempt 2 alone: a shot there under the planned rule would put those at risk, against a shot
  rate measured only below 0.5.
- On a scored run of about 240 tasks, below 0.5 holds about 68 tasks at `P2.6-01`'s share (34 of
  120) and 38 at `P1.1-01`'s (19 of 120), more than the about 15 the last 3 hours afford, so the
  ceiling rarely binds; the order (least sure first) picks the tasks.

**Recommendation:** keep `max_confidence` 0.5 and the planned merge rule ($N_1$ kept as one of the
two attempts whenever NVARC has a real candidate, $S_1$ in the other slot), since below 0.5 NVARC's
attempt 2 never held a right answer and the model gives no second answer of its own.
