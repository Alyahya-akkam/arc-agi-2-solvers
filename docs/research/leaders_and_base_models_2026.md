# Leaders' methods and base-model alternatives for the last 24 days

Author: `researcher` subagent. Date: 2026-10-09. Feeds: the human's question on BRIEF B2 Q5 (base
model fixed for 2026), the plan for the remaining days to the 2 Nov deadline.

## Question

What are the top ARC Prize 2026 (ARC-AGI-2) Kaggle entries and the ARC Prize 2025 winners doing
differently from NVARC's recipe, and is there a publicly available, offline-usable base checkpoint
(dense up to about 14B, or MoE that fits 4x L4) that would plausibly beat
`sorokin/qwen3_4b_grids15_sft139` in our setting within the remaining 24 days?

## Sources and conventions

| Source | Revision or read time | Kind |
| --- | --- | --- |
| Kaggle web endpoints the site's JavaScript calls (`https://www.kaggle.com/api/i/<service>/<method>`, POST, anonymous session cookie, no login, no API token, no CLI): `competitions.LeaderboardService/GetLeaderboard` `{"competitionId":133469}`; `competitions.CompetitionService/GetCompetition`; `discussions.DiscussionsService/GetTopicListByForumId` (forum 10403419, pages 1 to 8, 90 topics); `discussions.DiscussionsService/GetForumTopicById` (34 topics); `kernels.KernelsService/ListKernels` (competition 133469, 524 public notebooks); `discussions.WriteUpsService/GetWriteUpBySlug` (2025 writeups) | 2026-10-09 07:14 UTC; leaderboard JSON sha256 prefix `6c25e85bff17da78` (scratch copy, not committed) | primary (Kaggle) for scores, counts, dates; forum posts are participant statements |
| Leaderboard history monitor by Tong Hui Kang, page `https://arc3.huikang.dev/leaderboard/?comp=arc2`, data `https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc2` (announced in forum topic 709522) | 2026-10-09; sha256 prefix `3dbe33745ca01f29` | third party |
| arcprize.org `/media/data/evaluations.json`, `/media/data/models.json` (the data behind `/leaderboard`), `/results/alibaba-qwen3-8-27b`, `/competitions/2025`, `/competitions/2026/arc-agi-2`, `/blog/arc-prize-2026-milestone-1` | read 2026-10-09 | primary (organizers) |
| ARC Prize 2025 Technical Report, arXiv 2601.10904v1 | read 2026-10-09 | primary (organizers) |
| `github.com/1ytic/NVARC` `nvarc_2025.pdf` (11 pages) | `846d0198efa752534594e321fc3289fc0a06c657` | primary (authors) |
| ARChitects 2025 report, `https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/` | read 2026-10-09 | primary (authors) |
| Kaggle writeups `arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution` and `.../nvarc` | read 2026-10-09 | primary (authors) |
| Hugging Face Hub API `https://huggingface.co/api/models/<id>` and model cards; revisions cited inline | read 2026-10-09 | primary (publishers) |
| This repo: `RESULTS.md`, `docs/experiments/P1.1-01.md`, `docs/research/kaggle_first_runs.md`, `docs/BRIEF.md` B2 Q5, B3 | worktree `wt-W40-orch`, 2026-10-09 | project |

Kaggle page bodies stay unreadable to a plain fetch (as in `kaggle_first_runs.md`), but the JSON
endpoints above return the same data the pages render. Notebook sources were not readable
anonymously (`GetKernelVersion` returns 403), so notebook contents are known by title and attached
inputs only. Forum statements are quoted as what a named participant said, not as facts; anything
resting on them, on search snippets, or on third parties is marked **unverified**.

## Short answer

- **Leaderboard (public, 50% of the hidden test):** Tufa Labs 88.06, rabbithole 80.56,
  Yi-Chia Chen 77.22, nvbanana (CPMP and Darragh, last year's NVARC) 77.08, then 67.64, 67.36,
  58.75, 58.47. Prize places (top 8) start at 58.47, gold at 38.47 (rank 15). We are rank 1061 of
  2638 with 29.72. No top team has published its 2026 method.
- **Three differences from our recipe** (inferred from organizer data, team statements, and the
  score timeline, not from published methods):
  1. New model weights. The best public notebook scores 33.89 and every public notebook at 31.8
     or more runs our checkpoint; nvbanana says simple tweaks to it "won't be enough" and its
     recent submissions are "same code as before, just a new checkpoint".
  2. Long-reasoning models that write and run code (induction), next to or instead of direct grid
     transduction. Qwen3.8-27B (Apache 2.0, 27.8B dense) scores 42.4% on ARC-AGI-2 semi-private at
     XHigh effort and 1.5% with thinking off; CPMP: "this model reshuffled everything". NVIDIA
     published 207k ARC agentic SFT traces with a Python executor tool.
  3. Inference engineering for 64k-token generations on 4x L4 (FP8 weights and KV cache, MTP or
     EAGLE speculative decoding, SGLang or vLLM, forced commit), not Unsloth HF decoding.
- **Drop-in checkpoint:** none. No public ARC-trained checkpoint beats ours (its 2B sibling, TRM,
  MindsAI's CodeT5 and the ARChitects' 2024 models all score lower), and general models without
  ARC SFT score 1 to 2% when used as direct transducers. The only public model in our size class
  with a verified ARC-AGI-2 score above ours is Qwen3.8-27B, and only with long reasoning. It would
  be a second solver, not a swap of our base.
- **Recommendation:** keep the base (no swap: (b) has no candidate). Commit to (a): selection plus
  the work already planned. Run (c), adding Qwen3.8-27B as an execution-verified program-synthesis
  solver, only if the human waives BRIEF A0's "nothing textual" rule for it, and then as a 3-day
  feasibility spike with a hard gate. Details and day counts in Section 5.

## 1. The 2026 leaderboard and what the top teams have said

### 1.1 Competition facts (Kaggle `GetCompetition`, read 2026-10-09)

Deadline 2026-11-02 23:59 UTC; team-merger and new-entrant deadline 2026-10-26 23:59 UTC; public
notebook publishing disabled after 2026-10-26; 1 submission per day; 2 final selections;
leaderboard on 50% of the hidden test; 720 GPU minutes; 2637 teams, 23,663 submissions. The
accelerator is 4x L4 (96 GB), upgraded on 2026-04-07; H100s went to the ARC-AGI-3 track (forum
689054, Kaggle staff). CPMP states that the public leaderboard data is the semi-private set used
for the verified leaderboard on arcprize.org (forum 743880, **unverified**).

Queue state (forum 747515, Kaggle staff, 2026-10-08): for two days submissions waited more than
15 h before starting and were marked "Kaggle Error"; the queue timeout was extended, affected runs
are being scored, a team may now have only one submission pending at a time, and a Kaggle Error
does not count against the daily limit (forum 746520). Runtime counts running time only. Each
4x L4 hour costs two quota hours (forum 689054, participant Sam-wiz, **unverified**; matches
`kaggle_first_runs.md`).

### 1.2 Top of the public leaderboard (Kaggle `GetLeaderboard`, 2026-10-09 07:14 UTC)

| Rank | Team | Members | Score | Subs | Crossed 40 / 70 / 80 (history monitor, **third party**) |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | Tufa Labs | Jeroen Cottaar, Dries Smit, Stefano Viel, IsaiahP | 88.06 | 47 | first sub 08-22; 08-29 / 09-16 / 09-22; 85+ on 10-07 |
| 2 | rabbithole | Xie Zejian, Cookize Song, Yuhua Ke | 80.56 | 108 | 07-21 / 08-23 / 10-03 |
| 3 | Yi-Chia Chen | Yi-Chia Chen | 77.22 | 17 | 09-08 / 10-04 / no |
| 4 | nvbanana | CPMP, Darragh | 77.08 | 138 | 49.17 at first record (05-31) / 08-18 / no |
| 5 | _hans | _hans | 67.64 | 12 | 10-02 (from 28.06) |
| 6 | ALbert RUehlman | Albert Ruehlman | 67.36 | 7 | 10-03 |
| 7 | ARC Raiders | Cho Royou, NT, chiefkeith, 上原 歩夢 | 58.75 | 67 | 10-05 (33.06 to 42.64 to 58.75 on 10-07) |
| 8 | Dawid Kopiczko | Dawid Kopiczko | 58.47 | 31 | 10-04 |
| 9 | Majkel1337 | Majkel1337 | 48.33 | 34 | 10-03 |
| 10 | Khaotic Sethesis | Kha Vo, Seth Moudry | 45.83 | 153 | 10-01 |
| 11 | Anil Thomas | Anil Thomas | 44.31 | 19 | 10-03 (from 31.11) |
| 15 | Sam-wiz | Sam-wiz | 38.47 | 7 | last gold |
| 19 | Jack Cole (MindsAI) | Jack Cole | 36.94 | 181 | no |
| 1061 | Alyahya Akkam (us) | Alyahya Akkam | 29.72 | 2 | no |

Distribution: 8 teams in the prize places (58.47 and up), gold ranks 1 to 15 (38.47 and up),
silver to rank 131 (32.22), bronze to rank 263 (31.81); 29 teams at 33.89 or more; median 28.61;
1091 teams at 29.72 or more. The monitor's runtime column shows 600 to 800 (units undocumented,
consistent with minutes of near-full 12 h runs) for the top teams' recent runs.

Two waves stand out: the top four climbed through June to September; ranks 5 to 13 jumped from
the NVARC band (28 to 33) to 40 to 68 between 28 Sep and 7 Oct.

### 1.3 What top teams and active participants have said (forum, participant statements)

- CPMP (nvbanana), 2026-04-24, topic 691081: "simple tweaks to our last year winning solution
  won't be enough to win this time." 2026-08-02 (724647): "We won't share more till the
  deadline." 2026-10-08 (747088): "Same code as before, just a new checkpoint."
- CPMP, 2026-10-04 (745787): "We got 70+ before Qwen 3.8, and this model reshuffled everything."
- cm391, 2026-08-06 (724647), speculation: NVIDIA's `nvidia/Nemotron-SFT-ARC-AGI-v1` (Section 3)
  shows the tools (Python executor, commit, augmenter) nvbanana might use. CPMP: "I won't comment
  on the rest" (he confirmed only that the team name plays on "nano banana").
- Anil Thomas (rank 11), 2026-10-04 (745787): Qwen3.8 needs XHigh; generations have a median
  around 64k tokens; with thinking off "the results are abysmal"; a back-of-envelope full run at
  350 tokens/s on 4x L4 gives about 30. He reads scores above 45 as "something super-clever".
- NguyenThanhNhan, 2026-10-05 (745787): two submissions of Qwen3.8-27B FP8 with MTP through SGLang
  (FP8 KV cache, EAGLE 3 steps), 65k token cap, forced commit after 49k, one pass only, "pure
  reasoning". His team's best stayed 32.78 (leaderboard), so these runs scored at most that.
- juliencst, 2026-10-04 (745787): pure Qwen3.8 submissions "reaching scores of 8.xx".
- Cho Royou (ARC Raiders, rank 7), 2026-10-07 (742796): "QLoRA fine-tuning a Qwen 4B model might
  be a good option. Although the 27B model can run on four L4 GPUs, the cost is quite high."
- Dominic, 2026-08-04 (732723): two 35B-A3B Qwen3.5-MoE post-trains on 4x L4 (vLLM 0.26, FP8),
  438 and 452 tokens/s, 0 of 64 ARC-AGI-2 evaluation grids solved; W4A16 is 22.8 GiB.
- Adarsh, 2026-07-26 (729743): on NVARC, NLL-threshold changes, several selection strategies, a
  second decode round, TRM ensembling: "some helped in CV, couldn't translate to LB"; "the only
  significant way to boost score is to create more synthetic data and further train" the model.
- Run-to-run spread of identical NVARC code on this leaderboard: a verbatim fork 28.06 against
  31.53 and 31.39 for byte-identical or path-only copies (745846); copies at 29.72 and 29.31
  against the authors' 32.22 (742271). The NVARC paper (section 3.5) reports 1 to 2 points.
- adam99 (745846, **unverified**, method of measurement unclear): the hidden test's grids are
  sized like the training split (median max input area 184.5 against 551 for public eval; 7.1%
  multi-test tasks against 40.8%). If true, DEV120 overstates per-task cost.

Public notebooks (`ListKernels`, 524 scanned): the best public score is 33.89
(`koushikrudra/failed-in-aimo`), and every notebook at 31.8 or more attaches
`qwen3_4b_grids15_sft139`. Hybrids published so far do not beat the plain reproduction: NVARC plus
TRM 31.11, NVARC with Qwen3.8-27B FP8 attached 31.81 (`ghazarosbarseghyan91/fork-of-nvarc-plus-adaptive`;
whether it uses the 27B is unread),
NVARC plus Qwen2.5-Coder-7B induction 28.89; standalone: Qwen3.8-27B free-form 10.14
(`russcore/arcagi2-qwen38-freeform-direct`), TRM alone 10.0, gpt-oss-20b direct 2.92. A notebook
titled "arc-agi-2 inference qwen3.8-27b-fp8 within 4 hours" (`delongyang`, 2026-10-06) has no
public score; its content is unread.

No public writeup from ranks 1 to 8 was found (forum, Kaggle code, arcprize.org, web search). Tufa
Labs' ARC-AGI-3 Milestone 1 winner ("The Duck") runs "Qwen 3.6 27B FP8 locally" as an agent that
writes and runs Python in a REPL (arcprize.org `/blog/arc-prize-2026-milestone-1`); that is the
ARC-AGI-3 track on H100, so it shows the team's tooling, not its ARC-AGI-2 method.

## 2. ARC Prize 2025: results and the top entries

| Entry | Private score | Model | Data and training | Test time | Source |
| --- | ---: | --- | --- | --- | --- |
| NVARC (1st) | 24.03 (arcprize verified semi-private 27.6, $0.20/task) | Qwen3-4B-Thinking-2507, full fine-tune; 16-token grid vocabulary in the Qwen3 chat template | 103,253 synthetic puzzles (gpt-oss-120b, Claude) from 3,268 summaries, 3.2M augmented samples incl. "NVARC full" built from ARC-AGI-2 *evaluation* summaries; 4 nodes of 8x H100 for 27 h | LoRA r 256 per puzzle, batched DFS, 8-view rescoring `score_agg` (count plus mean log-probability); TRM ensemble gave no gain with the 4B (27.22 both ways) | NVARC pdf sections 2, 3.1 to 3.5, 4.4, Tables 1 and 2; tech report Table 1; arcprize `evaluations.json` |
| the ARChitects (2nd) | 16.53 (21.67 public) | LLaDA-8B masked diffusion with a multi-direction 2D RoPE | ReARC, ARC-GEN-100K, ARC-AGI-1 and 2, ARC-Heavy, ConceptARC; 175k steps, batch 8, 39 h on 8x H100 | rank-32 LoRA, 128 steps per task; 102 steps of soft-mask recursive refinement; most-visited selection | ARChitects 2025 report; tech report Table 1 |
| MindsAI & Tufa Labs (3rd) | 12.64 (tech report) or 15.42 (team writeup) | CodeT5-Large pruned to 660M (24 encoder, 16 decoder layers) | over 100M examples on TPUs, many months | TTT on about 45k examples, AIRV with 10k augmented inferences, two-checkpoint ensemble; about 11 h on 4x L4 | MindsAI writeup A3, A5, A8; tech report Table 1 |
| Paper award 1: TRM | 8% ARC-AGI-2 (report); 6.2% verified, $2.10/task | 7M recursive network plus puzzle embeddings | per-task embedding table; NVARC's Kaggle variant 10.0% in under 4 h | embedding re-init and fine-tune at test time | tech report; arcprize data; NVARC pdf 4.3 |
| Paper award 2: SOAR | n/a | LLM fine-tuned on its own evolutionary search traces | | evolutionary program synthesis | arcprize.org `/competitions/2025` |
| Paper award 3: CompressARC | 4% | 76K parameters, no pretraining | | per-puzzle MDL training | tech report |

Differences from our recipe that the 2025 entries show: the ARChitects changed the architecture
(diffusion, 2D positions, iterative refinement) rather than the data; MindsAI scaled TTT and
augmented inference by two orders of magnitude on a much smaller model; NVARC's gain came from
synthetic data scale. MindsAI's conclusion: "ARC-AGI-2 appears partially adversarial to the
current TTT/AIRV paradigm (at least with models this small), new ideas are needed." The organizers'
theme for 2025 is the refinement loop: per-task iterative program optimization guided by a
feedback signal (tech report, section on refinement loops).

## 3. Public ARC-specialized checkpoints

| Checkpoint | Size, license | ARC-AGI-2 evidence | Format and tokenizer | Cost on 4x L4 |
| --- | --- | --- | --- | --- |
| `sorokin/qwen3_4b_grids15_sft139` (ours, Kaggle Models) | 4B; license on the Kaggle page unread; CPMP enabled fine-tuning on 2026-08-13 (forum 735058) | 29.72 public 2025 (NVARC Table 2); 2026 public band about 28 to 34 (Section 1.3); ours 29.72 and 28.47 | 16-token grid vocabulary (NVARC 3.1) | 12 h for 240 tasks (Table 2); ours: P2.6-01 projects 0.74 of 12 h on 240 tasks (`RESULTS.md`) |
| `sorokin/qwen3_2b_grids15_sft141` (Kaggle Models) | 2B, LLM part of Qwen3-VL-2B-Instruct; fine-tuning enabled 2026-08-25 (735058) | 25 of 120 eval, 22.22 public in 6 h 10 min (NVARC Table 2); best 2026 public notebook with it alone 27.22 | same SFT data as the 4B per the paper; tokenizer identity with ours **unverified** | about half our time |
| TRM, `cpmpml/arc-prize-trm-031` (Kaggle dataset) | 7M plus embeddings | 10.0 public in under 4 h (NVARC 4.3); TRM alone 10.0 in 2026 | per-puzzle embeddings, own data format, own stack | 2 to 4 h |
| `mindware/arc-codet5-660m` (HF, rev `02488e59`) | 660M, MIT (card) | 12.64 or 15.42 private 2025 | CodeT5 BPE text format | about 11 h for a two-model ensemble |
| ARChitects 2025 LLaDA-8B | weights not published per their report; code on GitHub | 16.53 private | masked diffusion, own format | not reported |
| `da-fr/Mistral-NeMo-Minitron-8B-ARChitects-*` (HF, 2024) | 7.6B, NVIDIA open model license | ARChitects 2024 system: 2.5% verified on ARC-AGI-2 | ARC-AGI-1 era format | not measured |
| Trelis Qwen3-4B and gpt-oss-20b ARC-AGI-2 fine-tunes (HF, 2025), `asingh15/nvarc-*`, 2026 uploads (`Nabidnur/...`, `Omnisensai/...`, `ritwika96/...`) | various | no reported score found | various | n/a |
| `konstantinboyko/qwen3-4b-bfloat16-v02-01-01` (Kaggle Models) | 4B | notebooks with it score 29.86 and 30.97 | provenance **unverified**, likely a copy of ours | as ours |

Datasets, not models: `sorokin/nvarc-synthetic-puzzles` (Kaggle) and NVIDIA's
`nvidia/Nemotron-SFT-ARC-AGI-v1` (HF rev `92837449`, created 2026-05-26): multi-turn agentic
traces with a Python executor, a grid manipulator and a submit tool, filtered to exact matches;
67,529 traces (wide) and 207,569 (deep); teachers kimi-k25 (44.8%), qwen3-235b (34.5%),
deepseek-v3 (9.9%) and others; ARC-AGI evaluation IDs blacklisted; CC BY 4.0 per the summary
(card metadata says "pending-legal-review"). A companion `nvidia/Nemotron-RL-ARC-AGI-v1` exists.
No public model trained on them shows an ARC score.

Verdict: nothing public beats our checkpoint in a transduction pipeline. The 2B sibling is the
only drop-in with the same format, and it trades score for time.

## 4. General models at a size that fits

### 4.1 Verified ARC-AGI-2 scores of open-weight models (arcprize.org `evaluations.json`, semi-private)

| Model (effort) | Score | Size, license (HF API) | Fits 4x L4? |
| --- | ---: | --- | --- |
| Dots3-Note Preview (Max) | 76.8 | 288B, Apache 2.0 | no |
| DeepSeek V4.1 Flash (Max) | 72.9 | 763B, MIT | no |
| GLM-5.3-Flash (Max) | 65.8 | 321B, MIT | no |
| Kimi K3 (Max) | 60.4 | large (size not checked) | no |
| **Qwen3.8-27B (XHigh / Low / Medium / None)** | **42.4 / 22.8 / 13.2 / 1.5** | 27.8B dense (hybrid Gated DeltaNet and gated attention, 64 layers), Apache 2.0; official FP8 `Qwen/Qwen3.8-27B-FP8` rev `017b9c7a` | yes: about 28 GB of weights in FP8 (arithmetic) |
| Inkling Small (XHigh) | 40.1 | 266B, Apache 2.0 | no |
| Qwen3-235B-A22B Instruct 2507 (no reasoning) | 1.2 | 235B | no |
| NVARC 2025 (for reference) | 27.6 | 4B | yes |

No arcprize.org ARC-AGI-2 entry exists for the other models that fit: Qwen3.5-4B and 9B (Apache
2.0, Feb 2026), Qwen3.6-27B and Qwen3.6-35B-A3B (Apache 2.0, Apr 2026), NVIDIA Nemotron 3.5
Lightning 30B-A3B (OpenMDW 1.1, Aug 2026), gpt-oss-20b. Qwen3.8-Flash-Next (125B plus 51B n-gram
embeddings, 6B active) is under the Qwen Community License 1.0, whose prize eligibility is an open
forum question (746307); it has no verified score either. Qwen3.8-27B was tested on 2026-08-14 at
$0.45 per task (`/results/alibaba-qwen3-8-27b`); a participant estimates about 64k tokens per
query at XHigh (Section 1.3, **unverified**).

### 4.2 Why a general model does not work as the base of our pipeline

1. **No transduction prior without ARC SFT.** Our pipeline asks the model for the output grid
   directly, with no reasoning text, then searches (DFS with a probability cutoff) and ranks by
   augmentation agreement. General models in that mode score 1 to 2% (Qwen3.8-27B with thinking
   off 1.5, Qwen3-235B Instruct 1.2, DeepSeek V4 Flash with no effort 2.1). NVARC's whole edge is
   the 3.2M-sample SFT into a 16-token grid format (section 3.1, Figure 1: loss falls as synthetic
   data grows); 128 TTT steps per puzzle cannot replace it.
2. **Their strength needs long reasoning, which the pipeline does not use.** Qwen3.8-27B goes from
   1.5 to 42.4 only with XHigh thinking, at tens of thousands of tokens per query.
3. **Compute.** TTT and DFS scale with parameter count. Our 4B already projects to 0.74 to 0.91 of
   12 h on 240 tasks (P2.6-01, P1.1-01). A 9B costs about 2.4 times the FLOPs, a 27B about 7 times,
   and a 27B bf16 LoRA does not fit one 24 GB L4 (55 GB of weights), so per-puzzle TTT with one
   worker per GPU breaks.
4. **Tokenizer and parity.** A new base means a new grid tokenizer and format and new augmentation
   scoring calibration: `tokenize.py`, `decode.py`, and the P0.1 parity fixture would no longer
   apply, and H (Phase 3) is trained against our base's hidden size.

### 4.3 Cost of an ARC SFT on our compute

NVARC's full fine-tune of the 4B used 4 nodes of 8x H100 for 27 h, 864 H100-hours (NVARC 3.1); the
ARChitects' LLaDA-8B used 312 H100-hours (their report). Our budget is about 30 quota hours per
week, and 4x L4 bills at twice the rate, so about 15 h of 4x L4 (60 L4-hours) per week. An L4 is
roughly an order of magnitude slower than an H100 for bf16 training (spec-sheet ratio, not
measured here), so a NVARC-scale SFT is on the order of a hundred weeks of our quota, and even 1%
of it is about a week. Continued SFT on a small subset is affordable but has no evidence of gain:
a participant reports extra SFT from the NVARC checkpoint did not help on hard tasks because his
data was too easy (forum 733930, **unverified**).

## 5. Recommendation for the remaining 24 days

### 5.1 Options

**(b) Swap the base: no.** No public checkpoint in the size limit beats ours in our pipeline
(Sections 3 and 4). The 2B sibling would be a drop-in (same format per the NVARC paper; tokenizer
identity to be checked) but scores lower (22.22 against 29.72 in NVARC's own runs); it only buys
time.

**(a) Stay on the base, spend the effort on selection and the planned work.** Low risk, no new
stack, all the existing parity and timing work holds. Expected ceiling: the NVARC band, about 30 to
34 public (rank roughly 30 to 1000), because:
- the DEV120 oracle gap is 6.83 tasks (`P1.1-01`: 44.00 against 37.17 of 120), but DEV120 is
  contaminated for this base (NVARC Table 1 "NVARC full"), so the gap on the hidden test is
  unknown;
- our selection is already NVARC's improved `score_kgmon`/`score_agg`; Adarsh reports that other
  selection variants did not move the leaderboard (729743);
- identical NVARC code spreads about 28 to 34 on this leaderboard, so a 2 to 3 point real gain is
  hard to see with 1 submission per day.
Day count: selection variants on stored candidates are CPU work, 2 to 4 days plus 1 to 2
submissions.

**(c) Add Qwen3.8-27B as a second, execution-verified solver.** Keep NVARC for transduction;
spend the leftover time on Qwen3.8-27B FP8 writing a Python `transform(grid)` per task, executed on
the demonstration pairs; a program that reproduces every pair supplies an attempt (it is
high-precision because it is checked), the rest keep NVARC's attempts. This is the only change linked to large movement in 2026, and the link is circumstantial (CPMP's
statement that Qwen3.8 "reshuffled everything"; ranks 5 to 13 jumping from about 31 to 40 to 68
between 28 Sep and 7 Oct, methods unpublished; Qwen3.8 FP8 mirrors on Kaggle since early Sep). No
public notebook demonstrates the gain (NVARC plus Qwen3.8 public: 31.81; pure Qwen3.8 public: 8
to 10), so the size of the gain under our budget is **unverified**.
- What changes: a second inference stack (SGLang or vLLM wheels for the Qwen3.5 architecture,
  offline, in a separate environment from the Unsloth 2025.9.7 stack, run after the NVARC workers
  exit); a time split (free 3 to 4 h, e.g. by fewer decode views on easy tasks, or by targeting
  only tasks where NVARC's top candidates disagree); a sandbox to execute generated code; an
  attempt policy. `tokenize.py`, `decode.py`, and the parity fixture stay untouched.
- Budget arithmetic: at the 350 to 450 tokens/s participants report on 4x L4 (**unverified**),
  3 h gives 3.8M to 4.9M tokens, about 60 to 75 queries at 64k: roughly a quarter of the tasks.
- Conflicts and risks: BRIEF A0 says "Nothing in the pipeline is textual"; generated code is
  textual, so this needs the human's explicit waiver. Engine support for the hybrid DeltaNet
  architecture with FP8 on L4 (sm_89) inside Kaggle's image is **unverified**. Each 4x L4
  development hour costs two quota hours, and the queue currently delays submissions by many
  hours with one pending submission per team. DEV120 cannot measure the hybrid fairly (NVARC's
  half is contaminated; Qwen3.8's exposure to public ARC-AGI-2 data is unknown). Qwen3.8-27B is
  Apache 2.0, so it adds no license question beyond the one our base already has (Section 5.3).
- Day count: spike 3 days (offline engine on a 4x L4 development run: tokens/s and verified
  solves on about 20 tasks NVARC misses); gate on day 4 (go only if at least about 5 of 20 are
  solved and throughput is at least 300 tokens/s); integration and a full timing run 6 to 8 days;
  then 2 to 4 scored submissions in the last 8 to 10 days. Total about 12 to 14 days of work,
  ending around 23 to 25 Oct.

### 5.2 Recommendation

Keep `qwen3_4b_grids15_sft139` as the base for the whole entry; no swap. Commit to (a), and make
the final pair of selections from it unless (c) passes its gate. Ask the human one question now:
waive A0 for an execution-verified program-synthesis solver? If yes, run (c) as a 3-day spike in
parallel with (a), stop it at the gate if it misses, and if it passes, make one of the two final
selections the hybrid and the other the best pure-NVARC build. If no, (a) is the plan, with the
expectation that it stays in the NVARC band (about 30 to 34) and outside medal and prize ranges
(31.81 bronze, 38.47 gold, 58.47 prize on 9 Oct).

What to reuse: the NVARC transduction path and parity fixture as they are; `score_kgmon` candidate
dumps for offline selection work; the timing governor's per-op records to size the time split.
What to change: only additive stages behind a config switch, so the pure-NVARC build stays
submittable.

### 5.3 Open risks and unverified items

- No 2026 top-team method is published; the inferences in the short answer rest on statements by
  CPMP and others, on the score timeline, and on NVIDIA's datasets.
- Qwen3.8-27B throughput on 4x L4, tokens per query, and engine compatibility rest on forum posts.
- Whether the 2026 public leaderboard is the semi-private set (CPMP, unverified), and the hidden
  test's task sizes (adam99, unverified).
- MindsAI's 2025 private score differs between the organizers' report (12.64) and the team's
  writeup (15.42).
- The license of `sorokin/qwen3_4b_grids15_sft139` on its Kaggle page and the prize-eligibility
  reading of rules section 2.5 for non-OSAID base models (forum 740268, unanswered) apply to us
  whichever option is chosen.
