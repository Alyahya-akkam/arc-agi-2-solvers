# ADR-012: Meta-training sources and leakage gates depart from B5 P3.4 in four ways

- Status: Accepted, 2026-10-01. Source: task brief `docs/tasks/P3.4.md` (phase 2) and the research
  note `docs/research/meta_training_sources.md` ("the note").
- Tasks: P3.4 (episodes and leakage check), P3.5 (training reads the episodes), P3.7 (validation
  puzzles).

## Context

B5 P3.4 (`docs/BRIEF.md`, line 589) specifies: "sources ReARC (generator run offline), BARC
(dataset), ARC-AGI-1 and ARC-AGI-2 public training sets [...] Validation split by task ID with 200
held-out generated puzzles. Leakage check: no episode grid pair may hash-match any ARC-AGI-2
evaluation pair." The note measured the sources against each other and against the evaluation
IDs, and found four places where that wording either adds nothing or leaves a hole:

1. **ARC-AGI-1.** ARC-AGI-2 training already holds 391 of the 400 ARC-AGI-1 training tasks and 376
   of its 400 evaluation tasks with identical pair sets. Six of the other 24 ARC-AGI-1 evaluation
   tasks *are* ARC-AGI-2 evaluation tasks, three of them in DEV4 (note, short answer and 1.3).
2. **ReARC generators.** ReARC's `re_arc.zip` is the generators' own output: 1000 verified
   examples per generator at seed 42. The first pool uses about 285 per generator (note 4.4), and
   running the generators means vendoring `generators.py`, `verifiers.py`, and `dsl.py` for about
   0.6 CPU-hours (note 3.3). The same repository's `arc_original.zip` holds ARC-AGI-1 evaluation,
   the six DEV120 tasks included (note 1.3).
3. **Split unit.** A ReARC generator is written for the ARC-AGI-1 training task of the same ID, and
   391 of those tasks are in ARC-AGI-2 training. A split by task ID alone can put a generator in
   validation and its ARC task in training, or the reverse (note 4.5).
4. **Exact hashes.** Training augments every pair under $D_4 \times S_{10}$; third-party sets ship
   augmented copies of the evaluation tasks (NVARC's `arc2_evaluation6`); and ARC-AGI-1 itself has
   color-permuted duplicates across its split (`caa06a1f`, `50a16a69`): the exact hash finds 3
   shared pairs where the canonical one finds 5 (note 2.2). The evaluation solutions may be read
   only for this check (A0), and only on Kaggle.

## Decision

1. **No separate ARC-AGI-1 source.** Its tasks enter through ARC-AGI-2 training (767 tasks) and
   ReARC (400 generators). Nothing fetches the ARC-AGI-1 repository, the evaluation directory, or
   `arc_original.zip` (gate G0).
2. **ReARC from `re_arc.zip`, not by running the generators.** The zip is pinned by commit, git
   blob ID, size, and sha256 (`configs/episodes.yaml`); episodes take disjoint examples per
   generator.
3. **Validation split by family.** A family is a ReARC generator together with the ARC task of
   the same ID, or one BARC problem. 20 generator families are held out (the 9 ARC-AGI-1 training
   tasks that ARC-AGI-2 dropped, plus 11 seeded picks among non-BARC-seed generators), 5 puzzles
   each, and 100 BARC problems, one puzzle each: 200 puzzles whose shapes follow DEV120's joint
   counts. Then every training task with a pair canonically equal to a validation pair is dropped,
   and the build fails if any family or canonical pair still crosses the split.
4. **Leakage gates beyond an exact pair hash**, in a separate checker notebook with internet off
   (the builder has internet and no competition data):
   - G1: no source task ID, family, or BARC seed ID is an evaluation task ID.
   - G2: the exact pair hash $H_e$ matches no evaluation demonstration or test pair.
   - G3: neither does the canonical pair hash
     $H_c(p) = \mathrm{sha256}(\min_{g \in D_4} \mathrm{ser}(\rho(g x, g y)))$, which is equal
     exactly for $D_4 \times S_{10}$ images of a pair.
   - G4 and G5: input and output grids of at least 25 cells and 2 colors match no evaluation test
     input or test output, exactly or canonically.
   - A positive control injects 3 evaluation test pairs under a random dihedral map and color
     permutation; G3 must catch all 3.

   Every hit enters the report's drop list (a BARC seed as `seed:<id>`, which drops every
   problem that names it). The list is pasted into the config, and the episodes are rebuilt and
   re-checked until the check passes. Only IDs and counts leave the checker.

## Alternatives considered

1. **ARC-AGI-1 as its own source, as B5 says.** Lost: it duplicates 767 tasks already in ARC-AGI-2
   training, and its evaluation files carry six DEV120 tasks. Its 18 remaining tasks would add
   1.8% to the 989 ARC tasks, through files we must never fetch (note 1.3).
2. **Run ReARC's generators on Kaggle.** Lost: three and a half times more examples than the
   first pool needs already exist, verified. It would cost vendored code and CPU time and add a
   failure surface. It stays the route to more data or another difficulty (note 3.3).
3. **Split by task ID, as B5 says.** Lost: it can leak a generator's concept across the split
   through the same-ID ARC task. A family split costs only the 11 ARC-AGI-2 training tasks of the
   held-out generators (989 of 1000 remain; note 4.5).
4. **An exact pair hash only, as B5 says.** Lost: it misses every augmented copy, the case
   training itself creates and third-party sets ship (note 2.2). Each further choice also lost:
   - Exact checks of every training-time augmentation: $8 \cdot 10!$ variants per pair, against
     one canonical hash.
   - Grid gates without floors: 38 exact and 681 canonical false positives at 9 cells against
     unrelated tasks (note 2.3).
   - The check inside the builder: that notebook has internet, so it may not read evaluation data
     (G0, note 3.1).
   - No positive control: a check that has never caught a planted leak proves nothing.

## Consequences

- The sources are ReARC, BARC (the first 100k set), and ARC-AGI-2 training, all pinned by commit
  or revision plus sha256 or git object IDs. The 18 ARC-AGI-1 evaluation-only tasks are unused.
- ReARC diversity is bounded by 1000 examples per generator at the zip's difficulty. Generator
  code is not vendored, so `generators.py` is not part of our license surface.
- The BARC half of validation is a concept-level test: held-out problems share seeds with
  training. The ReARC half is the clean unseen-generator test.
- A check costs two Kaggle CPU sessions and returns IDs and counts. A failed check costs a
  rebuild; the grid gates should drop on the order of 20 source tasks (estimate, note 2.3).
- P3.5 binds to the manifest sha256 of a build whose report has `pass` true.
- Residual risk, not hash-detectable: pixel-edited evaluation pairs, crops, scalings, and concept
  overlap (note 2.2). The base model's own exposure to DEV120 concepts through NVARC's synthetic
  data is fixed (note 4.6).
- These departures are listed under "What to change" in the note; BRIEF B5 P3.4 is not edited
  here.

## Overturn criterion

1. Add ARC-AGI-1 tasks only if the per-source loss measurement (note 4.6) calls for more
   human-written tasks than ARC-AGI-2 training has, and only from a fetch that never downloads
   the evaluation directory.
2. Run the generators (vendored at `e5b7f1d`, seeded per generator by sha256 of its ID) once a run
   needs more than 1000 examples per generator, or larger grids: open risk 2 of the note
   (DEV120 is large-grid).
3. Move to a finer unit than the family (for example concept clusters) if P3.7's ReARC and BARC
   halves disagree in a way that concept-level overlap explains. Never return to a task-ID split.
4. Raise the grid-gate floors if G4 and G5 drop more than 1% of any source's tasks. Revisit the
   canonical gate if the positive control ever fails, or a tested collision appears. Return to
   the exact hash alone only if the canonical hash is shown unsound, which the tests rule out
   today.
