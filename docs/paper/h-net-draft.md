# Demonstrations to Weights: A Hypernetwork Start for Test-Time Training on ARC-AGI-2

> A 135M-parameter hypernetwork reads a task's demonstrations and emits a rank-16 adapter; merged into the base model, it ties plain test-time training task for task and leads it early in training.

<!--
Draft 1, 4 Nov 2026, ARC Prize 2026 Paper Track. Built by build_pdf.py into h-net-draft.pdf.
Word limit 1,500 including captions; title, subtitle and references excluded (build_pdf.py --count).
Every number carries its source record in a comment. Branches: P3.9-02, P3.9-01, MT-03, ADRs on
task/P3.9b, task/P3.9 and task/MT-02r2; P1.1-01, P1.3-01 on main; P1.4 on task/P1.4; v19 logs in
~/wt-P5.6/runs/H-submit-01-v19/. Citations: [@key] resolves to refs.bib, numbered by first use.
-->

## Abstract

We introduce H, a 135M-parameter hypernetwork that maps an ARC-AGI-2 task's demonstration pairs to a rank-16 LoRA adapter for NVARC's 4B grid model in one forward pass. Test-time training (TTT) drives the strongest ARC-AGI-2 systems, yet starts every task from the same weights. H encodes demonstrations with the frozen solver itself and is meta-trained on BARC and ReARC episodes, ending with first-order MAML. On 40 public evaluation tasks, H alone solves 5.0 tasks against the base's 4.0, within noise; merged into the base before TTT, it ties plain TTT on every task (10.5 each) with lower training loss through step 45 of 128. The linked submission scores [PLACEHOLDER: v19 public and private scores].

<!-- src: H 134,958,208 parameters (v19 h_checkpoint.json; docs/modules/hypernet.md). DEV40 arms (a) 4.0, (b) 5.0, (c) 10.5, (g) 10.5; (g) below (c) through step 45 (docs/experiments/P3.9-02.md). -->

## 1 Introduction

ARC-AGI-2 [@chollet2025arcagi2] gives each task a few demonstration pairs of colored grids and asks for the outputs of new test inputs. Tasks are new, so winning systems adapt weights per task: they fine-tune a LoRA adapter [@hu2021lora] on augmented demonstrations before sampling answers [@akyurek2024ttt; @franzen2025poe; @nvarc]. In NVARC's pipeline, this test-time training (TTT) takes 197 s of a task's 622 s on one L4 GPU under a 12-hour budget, and restarts from the same weights on every task.

<!-- src: P1.1-01 (main): reference pipeline on DEV120, 4xL4, per puzzle train mean 197.2 s, total mean 622 s, train 30.7% of worker time. BRIEF B3: 12 h, four workers, one L4 each. -->

We ask whether this adaptation can be amortized: learn once, across many tasks, a map from demonstrations to useful weights, and pay one forward pass per task. Our contributions:

- H, a hypernetwork that reuses the frozen solver as its encoder and emits rank-16 factors for every attention projection.
- Meta-training on task loss, then a first-order MAML stage that trains H's output as a fine-tuning start.
- A paired five-arm evaluation separating H alone, H as TTT's starting factors, and H merged before TTT.

## 2 Prior work

**TTT for ARC.** Akyürek et al. [@akyurek2024ttt] showed that TTT on a task's in-context examples raises ARC accuracy up to sixfold over fine-tuned baselines. The ARChitects [@franzen2025poe] apply task-specific augmentations in training, depth-first sampling and scoring, with the model's own probabilities selecting answers. NVARC [@nvarc], first in ARC Prize 2025, added synthetic puzzles, a Qwen3-4B grid model and per-task LoRA at rank 256.

**Hypernetworks and generated adapters.** Ha et al. [@ha2016hypernetworks] generate one network's weights with another. Text-to-LoRA (T2L) [@charakorn2025t2l] builds a LoRA adapter from a task description in one forward pass, and Doc-to-LoRA (D2L) [@charakorn2026d2l] builds one from a long context. We borrow T2L's head design and D2L's injection of externally supplied factors. HyperDreamBooth [@ruiz2024hyperdreambooth] composes predicted weights into a diffusion model, then fine-tunes briefly; our merged arm follows that pattern.

**Learned starts.** MAML [@finn2017maml] and its first-order variants [@nichol2018fomaml] train an initialization for fast fine-tuning; MMAML [@vuorio2019mmaml] modulates that initialization per task.

New here: H reads demonstration grids, not text; its encoder is the frozen solver itself; and its output is merged into the base before unchanged TTT.

## 3 Approach

**Base pipeline.** NVARC's Qwen3-4B model reads and writes grids in a 16-token vocabulary [@nvarc]. Per task, TTT trains a LoRA for one epoch over 128 augmented views of the demonstrations (128 steps); depth-first sampling with augmentation rescoring then picks two attempts per test input. Our development runs train rank 16 on the q, k, v, o projections, which scored no worse than NVARC's rank 256 on our 40-task subset (10.5 against 8.5 tasks).

<!-- src: configs/P3.9-02.yaml (task/P3.9b): lora_r 16, targets q,k,v,o; 128 steps per fit (P3.9-02). P1.3-01 via ADR-002 gate G2: rank 256 all modules 8.5/40, rank 16 q,k,v,o 10.5/40, +2.0 at the noise floor. NVARC writeup sec. 3: 16 tokens, r = 256. -->

**Encoder.** Each demonstration pair passes through the frozen base under 8 augmentations. Mean-pooling the layer-24 hidden states (of 36) over output-grid positions gives $e_i \in \mathbb{R}^{2560}$; averaging over pairs and views gives $\bar{z}$. Embeddings are cached (208,030 in meta-training).

<!-- src: ADR-001 (8 views amendment); MT-02 and MT-03: 208,030 cache hits. -->

**Hypernetwork.** Following T2L, H concatenates a code of $\bar{z}$ with layer and module embeddings, runs an MLP trunk, and applies one linear head per module type across all 36 layers. It emits factors $A$ ($16 \times d_{in}$) and $B$ ($d_{out} \times 16$) for q, k, v and o, 11.8M values in all, applied as $W' = W + sBA$ with rsLoRA scale $s = \alpha/\sqrt{r} = 8$ [@kalajdzievski2023rslora]. The $B$ head starts at zero, so the untrained H reproduces the base. H has 134,958,208 parameters.

<!-- src: docs/hypernet_design.md; ADR-002 (11,796,480 values, alpha 32, zero B head); docs/research/d2l_t2l_reading.md sec. 3; parameter count from docs/modules/hypernet.md and v19 h_checkpoint.json. -->

**Objective.** An episode holds one task's demonstrations $D$ and a held-out query pair $(x, y)$. With $D$ in context, H minimizes $\mathcal{L} = -\log p(y \mid x, D; \theta + \Delta\theta_H(D))$ over output-grid tokens, backpropagating through the frozen base into H.

<!-- src: ADR-006. -->

**Training.** The 99,674 training episodes come from 51,048 tasks generated by BARC [@li2024barc] and ReARC [@hodel2024rearc] or taken from ARC-AGI-2 training; a hash check against the ARC-AGI-2 evaluation set found no overlap. H trained 4,180 steps of 32 episodes on 4×L4, then 80 first-order MAML steps: two inner Adam steps on the demonstrations at rate $4 \times 10^{-4}$, then the query loss as the outer loss. Validation uses 286 episodes from 200 held-out tasks.

<!-- src: P3.4 (task/P3.4): 99,674 train episodes over 51,048 tasks, checker v2 PASS, 0 hits after one drop. MT-02 and MT-03 (task/MT-02r2): 32 episodes per step, stop at step 4,180. P3.9-01 attempt 2 (task/P3.9): 80 steps to 4,260, k = 2, inner lr 4e-4 picked by the probe. Validation: MT-02. -->

**Three uses.** (b) *H alone*: decode with H's adapter, no training. (f) *Factor start*: TTT starts from H's factors. (g) *Merge*: write $W + sBA$ into the bf16 base weights, run standard TTT from a fresh zero-initialized LoRA, and restore $W$ after the task. The linked submission (v19) runs (g) with NVARC's original rank-256 TTT.

<!-- src: ADR-019 amendment 1 (arm g); v19 title "H merge then original rank256 training" (STATE.md, 2 Nov, the human's screenshot); v19 dev run: 144 layers merged, restored, checkpoint step 4,260 (~/wt-P5.6/runs/H-submit-01-v19/run_logs/h-start/). -->

![Figure 1. Merge-then-train. The frozen base encodes the demonstrations; H maps the pooled embedding to a rank-16 adapter, which is merged into the base weights; standard TTT then fits a fresh LoRA, and sampling picks two attempts. Shaded steps are new and add 2.3 s per task.](figures/fig1_pipeline.png)

## 4 Results

**Setup.** DEV40 is a fixed sample of 40 tasks (55 test outputs) from the public ARC-AGI-2 evaluation set. All five arms ran on every task in one Kaggle 4×L4 session (24 Oct 2026, one seed). A task scores the share of its test outputs solved within two attempts; one task is 2.5 points, and differences under 2 tasks are noise.

<!-- src: BRIEF (DEV40 definition, metric); P3.9-02 header and result: 40 paired tasks, 55 outputs, 128 steps for every (c), (f), (g) fit, seed 0. -->

| Arm | Tasks of 40 | Attempt-1 exact of 55 | Seconds per task |
|---|---:|---:|---:|
| (a) Base, no adapter | 4.0 | 3 | 178.3 |
| (b) H alone | 5.0 | 3 | 160.9 |
| (c) Plain TTT | 10.5 | 8 | 481.9 |
| (f) TTT from H's factors | 9.5 | 8 | 456.9 |
| (g) Merge H, then TTT | 10.5 | 10 | 487.9 |

: Table 1. Five arms on DEV40, paired per task. Seconds are per task on one L4 GPU, all stages included.

<!-- src: P3.9-02 arms table; reports/P3.9-02/h_validation.md. -->

**Merging ties plain TTT task for task.** Arms (g) and (c) solve the same 11 outputs, 10.5 tasks each. (g) gets two more outputs right at attempt 1 (10 against 8), within noise.

**The factor start costs fit quality, not tasks.** (f) solves 9.5 tasks, $-1.0$ against (c) (90% paired interval $-3$ to $+1$): no difference. But its training loss rises above (c)'s from step 20 and ends higher on 38 of 40 tasks (median ratio 2.2). Merging removes this penalty: (g) reaches (f)'s final loss on all 40 tasks (median step 63) and ends below it on all 40 (median ratio 0.47).

**H alone** solves 5.0 tasks against the base's 4.0 (one gained, none lost) without training.

**Training loss** (Figure 2). Step 1 runs at learning rate 0, so its loss measures the start: 0.0916 for (g) against 0.0946 for (c), 3% lower. (g) stays below (c) through step 45; then the curves cross 11 times, ending at 0.0098 against 0.0094. Per task, (g) reaches (c)'s final loss at median step 95 of 128: not sooner.

<!-- src: P3.9-02 "Training loss": step-1 means, through step 45, 11 crossings, finals, per-task reach 37 of 40 at median 95; (f) above (c) from step 20, 38 of 40 at median ratio 2.2; (g) vs (f) 40 of 40, median 63, ratio 0.47. Reproduced from records-rank*.jsonl by build_pdf.py --extract. -->

**Cost.** H adds 2.3 s per task: encoding 1.9 s, H's forward pass 0.02 s, merging 0.34 s and restoring 0.04 s. Outside decoding, (g) runs within a second of (c).

<!-- src: P3.9-02 arms table and "Time per puzzle": encode 1.9 s is arm (b)'s (g reads its cache); (g) +6.0 s mean, median -2.4 s against (c), within a second outside the decode. -->

**Leaderboard.** v19 scores [PLACEHOLDER: public score] public and [PLACEHOLDER: private score] private. For context, NVARC's notebook, rerun verbatim, scores 30.97 public; leaderboard differences under 2 points are noise.

<!-- src: RESULTS.md on task/P1.4 (P1.4 calibration, 30.97 public); v19 still scoring at the close (STATE.md, 3 Nov). -->

![Figure 2. TTT training loss on DEV40 for plain TTT (c), TTT from H's factors (f) and merge-then-TTT (g): mean over 40 tasks, trailing mean over 8 steps, log scale. Lower panel: (g) divided by (c). (g) stays below (c) through step 45 (dotted line), then ties it; (f) rises above (c) from step 20.](figures/fig2_loss.png)

## 5 Conclusion

H is cheap and safe to merge: 2.3 s per task, every base weight restored bit for bit, and a bf16 rounding error of 3% of its delta. Its value shows early in training and is gone by step 128, where it solves exactly what plain TTT solves. Our reading, from one run: a fresh LoRA trained from zero erodes the start's lead within about 45 steps, so the per-task fit, not the start, decides what is solved. Starting from H's factors gives up LoRA's zero-initialized $B$ and fits about half as tightly; merging keeps it. In meta-training, H cut held-out loss on BARC episodes by 17% to 28% but sat above the no-adapter loss on ReARC at 18 of 20 validations; ReARC was 15% of the episode draws, and the base's loss on it was already 2.7 times lower than on BARC.

<!-- src: P3.9-02 (restored 40 of 40, rho median 0.034, (f) about half as tightly); MT-02 (BARC 17% to 27%, ReARC above zero at 4 of 5, ReARC 15% share, zero CE 0.0160 against 0.0429) and MT-03 (BARC 23% to 28%, ReARC above zero at 14 of 15, flat since step 1,000); ReARC sampling weight 0.15 (configs/train_hypernet_MT.yaml, task/MT-02r2). -->

**Limitations.** One seed, 40 tasks. DEV40 used rank-16 TTT; v19 uses NVARC's rank-256 TTT, untested with H at this scale. H's held-out loss was flat after step 1,000. Next: (g) against (c) at 32 and 64 training steps, where the early lead could become saved time; episodes closer to ReARC's distribution; a larger rank or a set encoder.

## References

<!-- Generated from refs.bib by build_pdf.py, in order of first citation. -->
