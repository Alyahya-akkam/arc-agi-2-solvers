# Run P6.10-02: MTP against the base by context (runs/P6.10-02)

**Row: Fault.** no engine samples; the pool's source None is not the base's 'engine log', or no cache_config converts KV use; the last band has 0 pairs, under 6: the base reference cannot be matched there

Base fit (tests/testdata/p6_5_03_engine_windows.json): $a = 0.04392$ s, $b = 3.3611e-08$ s per token, $c = -5.238e-04$ s per stream; 673 lines, rms 0.00633, max 0.01774, $n$ [11, 15], largest $S$ 1,086,023.

| Band (tokens) | Pairs | Seconds | Rate | Base rate | q | q, 15 running (pairs) | n | Mean context | Past base S | Out of reach | 8 to 10 streams: pairs, q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 to 16,384 | 0 | 0 | None | None | None | None (0) | None | None | 0 | 0 | 0, None |
| 16,384 to 32,768 | 0 | 0 | None | None | None | None (0) | None | None | 0 | 0 | 0, None |
| 32,768 to 49,152 | 0 | 0 | None | None | None | None (0) | None | None | 0 | 0 | 0, None |
| 49,152 to C | 0 | 0 | None | None | None | None (0) | None | None | 0 | 0 | 0, None |

KV pool None; most running None, most waiting None, preemptions None; drafts None, $\alpha$ None, $L$ None.

Calls: {'total': 0, 'errors': 0, 'cancelled_at_stop': 0}.

Engine log: 0 tracebacks, 0 ERROR lines, 0 NaN lines; during the run {'tracebacks': 0, 'errors': 0, 'tracebacks_after': 0, 'errors_after': 0}. Shot: {'queued': 15, 'waves': 0, 'shot_tasks': 0, 'used': 0, 'stopped': "the engine was not up (see run.json's problems)"}.
