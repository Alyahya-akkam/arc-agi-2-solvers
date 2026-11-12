# Meta-training throughput (B5 P3.6)

Measured by run `P3.6-01` (3 Oct 2026, 4xL4, `docs/experiments/P3.6-01.md`): P3.5's training step
(frozen bf16 base, H trained through the injected rank-16 adapter on q, k, v, o, non-reentrant
checkpointing per layer, FlashAttention 2, DDP over 4 L4s, one episode per GPU per micro step).

| Tokens per episode | 512 | 1024 | 2048 | 5120 | 8192 | 9728 |
|---|---:|---:|---:|---:|---:|---:|
| Micro step s (4 GPUs) | 0.60 | 0.91 | 1.74 | 5.10 | 9.01 | 11.10 |
| Peak reserved GiB (cache kept, as in training) | 10.10 | 10.10 | 10.10 | 11.47 | 12.69 | 13.58 |
| Headroom GiB of 22.03 | 11.59 | 11.59 | 11.59 | 10.22 | 8.99 | 8.10 |

- **The 8192-token cap fits** with 8.99 GiB (41%) of headroom and no allocator retries; 9728 fits
  too (8.10 GiB), so a longer cap is possible later. One process at 8192: 8.95 s, 12.33 GiB.
- **H's forward:** 3.61 ms in evaluation mode, 4.03 ms in training mode (bound 50 ms, B5 P3.3).
- **Meta-training step** (accumulation 8, 32 episodes, P3.4's length mix, mean 1369 tokens):
  **13.42 s**, calibrated on a measured mixed step (factor 1.017): 1.68 GPU-seconds per episode.
- **One pass over the 99,674 training episodes:** 3115 steps, 11.6 h of 4xL4: 23.2 quota hours if
  4xL4 costs twice the T4 rate, 11.6 if not (unconfirmed). Not included: encoding the
  demonstrations (about 0.5 GPU-seconds per distinct pair at 8 views, `P3.2-02`; about 200k pairs,
  estimate) and validations.

**Steps per week for P0.4:**

| Quota h per 4xL4 session h | Quota h per week | Session h | Steps | Passes |
|---:|---:|---:|---:|---:|
| 2 | 20 | 10.0 | 2683 | 0.86 |
| 2 | 30 | 15.0 | 4024 | 1.29 |
| 2 | 45 | 22.5 | 6036 | 1.94 |
| 1 | 20 | 20.0 | 5366 | 1.72 |
| 1 | 30 | 30.0 | 8049 | 2.58 |
| 1 | 45 | 45.0 | 12,073 | 3.88 |

`P3.5-02` (Borderline) bounds how much of this is useful: the toy runs' held-out gain peaked
before the end of training, so the meta-training brief selects H at its best validation point and
plans at most two passes.
