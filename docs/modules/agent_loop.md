# Modules `arc26/agent/` (P6.2): engine client, mock engine, agent loop, attempt selection

Task P6.2 of the agent solver (ADR-020). The contract is `arc26/agent/types.py`; the sandbox, the
verifier and the helper library are P6.1's; the GPU spike that drives this code is P6.3's. The
design is `docs/research/agent_library_design.md`, sections 2.1, 4, 5.2, 5.4 and 5.5.

## Purpose

A language model writes `transform(grid)` for an ARC task in a per-task Python REPL; every
program that defines `transform` is re-run by the verifier on all demonstration pairs; only
programs that reproduce every pair, run on every test input, and pass the guards become attempts.
This task builds the conversation around that: the prompts, the executor feedback, the turn loop
with its stop rules and budgets, an asyncio runner over many tasks, the engine client, a scripted
engine for CPU runs, and the choice of the two attempts per test input.

## Files and interfaces

| File | Main interfaces |
|---|---|
| `tasks.py` | `load_tasks(path, ids=None)`, `tasks_from_challenges`, `task_from_json`, `count_test_inputs` |
| `prompts.py` | `SYSTEM_PROMPT`, `DIRECT_SYSTEM_PROMPT` (run `P6.3-04`'s direct answers: no REPL, answer blocks), `system_prompt(index=None, with_index=True)`, `helper_index()`, `task_text`, `task_message(task, prior)`, `initial_messages` |
| `views.py` | `View(ops, colors)` (`forward`, `inverse`, `task`, `key`), `fixed_view(ops, seed)`: a dihedral transform and a color permutation fixing 0, NVARC's operations, for `P6.3-04`'s memorization control |
| `feedback.py` | `grid_text`, `clip`, `verification_report`, `cell_report`, `truncated_note`, `compose_feedback`, `forcing_message`, `forcing_chars` |
| `loop.py` | `Budget`, `run_attempt(task, engine, repl, verifier, budget, ...)`, `run_tasks(tasks, engine, repl_factory, verifier, budget, ...)`, `fenced_blocks`, `parse_answer`, `program_source`, `plain`, `AttemptContext`, `attempt_context`, `current_attempt`, `append_record`, `read_records` |
| `engine.py` | `EngineConfig`, `OpenAIEngine`, `EngineError`, `parse_completion`, `QWEN38_EFFORT_NOTES` |
| `mock_engine.py` | `ScriptedEngine`, `ScriptedReply`, reply builders (`code_reply`, `answer_reply`, `no_code_reply`, `overlong_reply`, `error_reply`), `hostile_programs(task)` |
| `select.py` | `rank_candidates`, `build_selection`, `build_submission`, `write_submission`, `select_files`, `load_nvarc`, `program_size` |

`tasks.py` reads the Kaggle challenges format and never a solutions file: a file whose name
contains `solution` is refused, and a test entry's `output` is ignored.

## Prompt

- **System prompt** (about 2,400 characters, one fixed prefix for every task, so the engine's
  prefix cache holds it once): the task format; the REPL (`train` as `(input, output)` numpy
  pairs, `test_inputs`, `np`, `lib`); the program contract (self-contained, imports only from the
  allowlist numpy, scipy, math, itertools, collections, functools, copy, typing,
  `arc26.agent.lib`; `transform(grid)` gets a 2-D numpy array of ints (int64, as P6.1's verifier
  passes it) and returns a 2-D numpy array or a list of lists of ints 0 to 9; 2 s per call; no
  stored outputs, no dispatch on example sizes); what the check reports and that
  a verified program ends the task; the direct-answer block; a note to keep reasoning focused.
  When `arc26.agent.lib` imports, its `index_text()` follows under "Helper library"; when the
  module does not exist the index is omitted, and a library that exists but fails to import
  raises. `system_prompt(with_index=False)` is the no-library arm of the ablation (design 5.7).
- **Task message:** `Task <id>: n training examples, m test inputs.`, then
  `Example i input (HxW):` and `Example i output (HxW):` with the grids as rows of digits (Qwen's
  tokenizer splits digits one by one, so a row of $w$ cells costs $w + 1$ tokens), then
  `Test input j (HxW):`. Indices count from 0, as `train[i]` and `test_inputs[j]` in the REPL. A
  later phase can pass `prior` verifications: the last 2 are appended with their reports.

## A turn

1. The engine replies. Code blocks are the fences tagged `python`, `py`, `python3`, or untagged;
   an unterminated fence (a reply cut at `max_tokens`) is ignored.
2. Direct answers are fences tagged `answer` holding JSON, `{"test_index": j, "grid": [[...]]}`
   or a bare grid for test input 0. They are parsed in the loop, not in the REPL, so they need
   nothing from the sandbox and survive a REPL restart. Valid ones go to
   `AttemptRecord.direct_answers`; invalid ones (also blocks over 100,000 characters or nested
   too deeply for `json`) are reported back.
3. Up to `max_cells` code blocks run in the REPL in order, each with `cell_timeout_s`; the
   deadline is checked between blocks. REPL and verifier calls run in a thread pool, so the
   event loop keeps serving the other attempts while they block.
4. A block whose `CellResult.defines_transform` is true is verified. The verified source is
   `program_source(block)`: its imports (also a `try` whose body only imports, with its
   handlers), function and class definitions (with decorators), and assignments whose value
   reads none of `train`, `test_inputs`, `transform`, as the original lines. Prints, loops and
   calls that test the program in the same block run only in the REPL, so a model that defines
   and tests `transform` in one block is not failed for it. A source seen before in the attempt
   is not verified again. The verifier's result is converted to plain Python values (`plain`:
   numpy scalars and arrays to Python scalars and lists; an output that is not a valid grid
   becomes None) before it is reported or recorded.
5. A verified program ends the attempt at once (later blocks of the reply are not run). Else the
   feedback message is built and the next turn starts.
6. **Forcing turn** (`Budget.forcing_tokens`; run `P6.3-02`, revision 5). When a thinking reply
   ends at its cap (`finish_reason == "length"`) and verified nothing, the next request is a
   forcing turn: the user message is `forcing_message` (the cut reply's cell reports, if any,
   clipped to 1,500 characters, then "Your reasoning reached the limit of $m$ tokens and was cut
   off. Stop reasoning now. Reply with exactly one ```python block that defines transform(grid)
   ... and nothing else: no exploration, no prints, no tests, no prose. ..."; firmer since
   revision 6, after one of revision 5's three forced replies explored), sent with `thinking=False` for that call only, so the chat template prefills an empty
   think block and the model writes code at once, with the cut reasoning in its context when
   `preserve_thinking` keeps it. Its reply's cap is `forcing_tokens`; its blocks run and verify
   like any other reply's. At most one forcing turn per cut reply: a forcing reply that is cut
   too, or has no program, goes on under the usual rules (the truncation note or the no-code note,
   then a thinking turn if rounds and room are left). Forcing turns do not count toward
   `max_turns`; the record counts them (`AttemptRecord.forcing_turns`) and those whose reply
   defined `transform` (`forced_programs`), and the transcript marks both messages
   (`"forcing": true`). Why: revision 4 of that run ended 156 of its 201 engine calls at the cap
   while the model was still reasoning, and 3 programs came out of 2.43M tokens.

## Feedback format

The user message after a reply, capped at `feedback_chars` (default 6,000) for its body:

```text
Round 2/3. About 11234 tokens left for this task.

[cell 0] output:
(3, 3) (3, 3)
[cell 0] Program check: 1/3 training examples passed.
Example 0: exact_match=True
Example 1: exact_match=False, shape 4x4 (as expected), cell_accuracy=0.88
  first mismatching rows (yours | expected):
  row 1: 0120 | 0110
Example 2: exact_match=False, shape 2x5 (expected 5x2), cell_accuracy=0.00
  your output starts:
  01200
Test input 0: output 3x3
012
120
201
Verdict: not accepted.

[answer] recorded for test input 0.
```

It follows the executor of NVIDIA's ARC traces ("Results: k/n training examples passed", round
counter): per example the exact match, shape and cell accuracy; up to 3 mismatching rows (yours,
then expected) when shapes match, the first rows of the output when they do not; one line when
every example fails with the same error; the outputs on the test inputs; guard findings
(`rule (rejects the program)` or `rule (ranks it lower)`); a verdict. Cell output is clipped to
1,500 characters (head and tail kept around a `[... n characters cut ...]` marker), a report to
3,000. A REPL timeout, memory error or restart is named. A reply cut at `max_tokens` gets a note
to send code sooner; a reply without code or answers gets a nudge.

## Stop rules and budgets

`Budget` holds an attempt's limits and sampling settings: `max_turns` (replies, default 3),
`attempt_tokens` (generated tokens over all replies, reasoning included, default 12,000),
`turn_tokens` (one reply's `max_tokens`, default 8,000), `min_turn_tokens` (256),
`context_tokens` (the engine's `max_model_len`, or None), `temperature` (1.0),
`reasoning_effort`, `cell_timeout_s` (10), `max_cells` (4), `feedback_chars`,
`preserve_thinking` (False), and `forcing_tokens` (None: no forcing turns; at least
`min_turn_tokens` when set). With `preserve_thinking`, each reply's reasoning stays in the
conversation as `Message.reasoning`, and the engine sends it back with the earlier turns; it is
off by default because it lengthens every later prompt. Run `P6.3-02`'s revision 5 turns it on
with forcing turns (the preserve check of 12 Oct measured the round trip: turn 2's prompt grew by
the reasoning's count plus 4).

With $T$ = `turn_tokens`, $A$ = `attempt_tokens`, $W$ = `context_tokens`, $F$ = `forcing_tokens`
(0 without forcing), $C_t$ the generated tokens so far and $\hat{P}_t$ an upper bound of the
prompt, a thinking reply's `max_tokens` is

$$m_t = \min\left(T,\; A - C_t - F,\; W - \hat{P}_t - \rho\right), \qquad \rho = F + |f|_{\max} + 16 + 2 \cdot 16 + 64$$

(`Budget.forcing_reserve`, 0 without forcing; $|f|_{\max}$ = `forcing_chars()`, 1,813
characters, bounds the forcing message; $\rho$ = 6,021 at $F$ = 4,096), and a forcing reply's

$$m^f_t = \min\left(F,\; A - C_t,\; W - \hat{P}_t\right).$$

The first prompt's bound is $\min(\hat{P}^c_0,\; N_0 + 64)$: $\hat{P}^c_0$ one token per
character of its messages plus 16 per message and 64 per prompt for the template, $N_0$ the
server's count of the same request when the engine offers `count_prompt_tokens` (optional;
`OpenAIEngine` posts the rendered request to vLLM's `/tokenize`, `EngineConfig.tokenize_url`), 64
(`COUNT_SLACK`) for a template variable the count and the chat request could treat apart. A
count that fails or is not a positive integer leaves $\hat{P}^c_0$, so the bound is never looser
than the character bound. Why: the system prompt's prose has about 4 characters per token, and a
phase 2 opening that shows two failed programs with their reports lost most of its first turn to
the character bound (8,500 characters of them left no turn at all on DEV40's largest tasks).
After turn $t$, from the engine's own counts:

$$\hat{P}_{t+1} = P_t + g_t + |u_t| + 2 \cdot 16 + 64, \qquad g_t = \begin{cases} G_t + 16 & \text{reasoning kept} \\ \min(G_t + 16,\; |a_t|) & \text{otherwise} \end{cases}$$

with $P_t$ and $G_t$ the engine's prompt and generated counts of turn $t$ (the generated count
covers the reasoning and the text; 16 covers the think tags the template adds around a kept
reasoning), $|a_t|$ the characters of the visible text, and $|u_t|$ those of the next user
message (feedback or forcing message). Revision 4's bound added the kept reasoning at one token
per character: a 16k-token reasoning of about 60k characters would have left no room for any
later turn. If $m_t$ is below `min_turn_tokens` the attempt stops.

A cut thinking reply always gets its forcing turn: $G_t \le m_t$ gives $A - C_{t+1} \ge F$,
and $\hat{P}_{t+1} \le \hat{P}_t + m_t + 16 + |f|_{\max} + 96 \le W - F$, so $m^f_{t+1} = F$
unless the deadline ends the attempt first. The feedback header's "tokens left" is
$\min(A - C_t - F,\; W - \hat{P}_{t+1} - \rho)$ without the message itself.

| Stop reason | When |
|---|---|
| `verified` | a program reproduced every pair, ran on every test input, and no guard finding was fatal |
| `rounds` | `max_turns` thinking replies were received (and the last one's forcing turn, if it was cut) |
| `budget` | less than `min_turn_tokens` of room is left (tokens or context) |
| `engine_error` | the engine raised (any exception; logged with its traceback) |
| `deadline` | the run's deadline passed before a reply was requested or while it was awaited |

`turns` counts the replies received, forcing turns included; `prompt_tokens` and `completion_tokens` sum the engine's
counts over those replies. A failed or cut call adds nothing (its tokens are unknown). A REPL or
verifier exception is an infrastructure fault: it propagates and the attempt leaves no record.

## Runner

`run_tasks` keeps up to `concurrency` attempts in flight against one shared engine. Each attempt
gets a REPL from `repl_factory(task)` (closed afterwards) and the same `Budget`; attempt ids are
`<task_id>/p<phase>/<k>` for `k < attempts_per_task`, and each attempt's seed is
`attempt_seed(seed, attempt_id)` (sha256, 32 bits). The blocking REPL and verifier calls run in a
thread pool of `concurrency` threads, so the verifier must be thread-safe. Attempts not started by
the `deadline` (a `time.monotonic()` value) are skipped. Each finished attempt is appended to
`log_path` as one JSON line (open, write, close) of plain values (`record_to_json` converts numpy
values and refuses NaN and infinities); a partial last line from a killed run is terminated
before the next run appends, and `read_records` skips lines that do not parse. An
optional `transcript_path` gets one line per attempt with every message (replies with their
reasoning). `on_record` is called after each record is logged, for example to rewrite the
submission; its exceptions, a failed attempt, and a failed log write are logged and the run goes
on. An attempt that raises is logged with its traceback (`logger.exception` on
`arc26.agent.loop`) and passed to `on_error(attempt_id, exc)`, for example to count faults by kind
(the spike does); an exception in `on_error` is logged too. Records return in task order.

## Engine

`OpenAIEngine(EngineConfig(...), timing=TimingLogger(log_dir, "engine"))` posts non-streaming
requests to `<base_url>/chat/completions` through one `httpx.AsyncClient` per event loop (a
client is bound to the loop that made it, so a second `asyncio.run` with the same engine gets a
new one). At most `max_connections` (default 64) are in flight; none is kept idle, since a
generation lasts seconds and a localhost connection costs nothing measurable, so no socket
outlives its loop. Connect errors, connect timeouts and connections dropped before a status line
are retried with exponential backoff (`backoff_s` doubling to `backoff_max_s`, jitter 0.5 to 1);
HTTP errors, timeouts and malformed replies raise `EngineError` at once, since a retried
generation would run twice. Timeouts: connect (and write) 10 s; read and pool none by default
(`read_timeout_s`, `pool_timeout_s`), since a call waits for the whole generation and the loop's
deadline bounds it. Tokens come from `usage`; `text` from `message.content` (empty when null);
the reasoning from `message.reasoning_content` (SGLang) or `message.reasoning` (vLLM 0.31). A
server started without a reasoning parser leaves the reasoning in `content`: without a reasoning
field, a content holding `</think>` is split on the last one, so code is taken only from after
it. Each call appends `{puzzle_id, op: "engine", seconds, tokens}` (generated tokens) to the
timing log, with the task id from the attempt context (`attempt_context`, a `contextvars`
variable the loop sets, since `Engine.complete` takes no task id).

Qwen3.8's parameters, from the chat template and card of `Qwen/Qwen3.8-27B` (rev `1d4bf0f2`; the
template of `RedHatAI/Qwen3.8-27B-INT4` rev `91bd022d` is identical) and the sources of vLLM
0.31.0 and SGLang 0.5.21:

| Setting | Name and values | How it is sent |
|---|---|---|
| Thinking switch | `enable_thinking`, default on; off prefills an empty think block (the generation prompt ends `<think>\n\n</think>\n\n`) | `chat_template_kwargs` (`EngineConfig.thinking` for every call, `complete(thinking=...)` for one call, the loop's forcing turn; `thinking_kwarg`) |
| Effort | `reasoning_effort`: `xhigh` (default), `medium`, `low`; the template raises on any other value while thinking is on, and adds its instruction to the system message only for `low` and `xhigh`; with thinking off it reads neither (both sit inside the template's `enable_thinking` test), so it cannot raise | top-level field (`effort_transport="request"`, the card's way; vLLM 0.31 and SGLang 0.5.21 forward it into the template, and vLLM's `"none"` turns thinking off) or `chat_template_kwargs` (`"template"`, the vLLM recipe's way). On a call whose own switch is off the effort is not sent; its instruction (`EngineConfig.effort_notes`, the template's text in `QWEN38_EFFORT_NOTES`) heads the system message instead, so the rendered system block, and with it the whole earlier conversation, is the thinking turns' own and vLLM's prefix cache serves the forcing turn (else the first tokens differ and the whole conversation, up to 30k tokens, is prefilled again) |
| Preserved thinking | `preserve_thinking`, default on: earlier assistant turns are re-rendered with their `reasoning_content`, trimmed, as `<think>\n...\n</think>\n\n` before the content, so a reasoning cut at the cap, which has no `</think>` of its own, is closed by the template; the test does not read `enable_thinking`, so it holds with thinking off too (the template read 12 Oct; vLLM 0.31's request path measured by the preserve check, 12 Oct: growth $R + 4$) | an assistant message's `Message.reasoning` is sent as `reasoning_content` (`reasoning_field`), and `chat_template_kwargs.preserve_thinking` (`preserve_kwarg`) is true exactly when an earlier turn carries reasoning; the loop keeps reasoning only under `Budget.preserve_thinking`, so by default earlier turns render as content only, not with empty think blocks that mimic the non-thinking mode |
| A reply with thinking off | no reasoning | if the server's reasoning parser still files the text as reasoning (one that treats output without `</think>` as thinking), `complete` moves it to the answer |
| Sampling (thinking) | temperature 1.0, top_p 0.95, top_k 20 | `generation_config.json`, applied by vLLM by default; `EngineConfig.sampling` overrides |

## Mock engine

`ScriptedEngine(scripts, fallback=..., chars_per_token=4.0, delay_s=0.0, timing=None)` replays
per attempt id, else per task id, one reply per turn (the number of assistant messages in the
conversation), and `fallback` (a reply without code) past the script's end. Generated tokens are
$\lceil |r| / 4 \rceil + \lceil |t| / 4 \rceil$ for reasoning $r$ and text $t$; a reply over
`max_tokens` is cut, reasoning first, and finishes with `length`. Prompt tokens are simulated the
same way. `ScriptedReply.error` raises `ScriptedEngineError`; `delay_s` waits. `calls` records
every request, with its `thinking` switch; a call with thinking off drops the scripted reasoning.
`count_prompt_tokens` returns the simulated prompt count (`prompt_tokens`), records the
conversation in `counts`, and raises when `count_error` is set (the fallback path). `hostile_programs(task)` returns `literal_lookup`, `shape_dispatch`, `input_hash`,
`imports_os`, `opens_file`, `dunder_import`, `infinite_loop`, `huge_allocation` sources for the
guards and the sandbox of P6.1; nothing in this module runs them.

## Attempt selection

Per test input (`rank_candidates`), following design section 4:

1. Verified programs are grouped by their output on that test input. Groups rank by votes (the
   distinct attempts behind them), then by their smallest program (AST nodes), then non-demoted
   first (demoted when every program behind the group has a non-fatal finding), then first seen.
2. When NVARC's first attempt equals a verified output, that group goes first.
3. With a verified group: attempt 1 the top group, attempt 2 the next group, else NVARC's first
   attempt if it differs, else the model's direct answer, else the best unverified program's
   output, else NVARC's second attempt (added as a last resort before `[[0]]`).
4. Without one: NVARC's two attempts (design section 4, "No verified program"), then direct
   answers, then unverified programs.

Direct answers rank by votes, then first seen. Unverified programs (not verified, no fatal
finding, a valid output on this test input) rank by mean cell accuracy over the demonstrations,
then pairs passed, then size. An NVARC grid equal to `[[0]]` is NVARC's placeholder for a slot
it did not fill (`ArcDataset.get_submission` pre-fills both attempts with it), so it counts as
missing. Grids from records are converted with `plain` (numpy arrays accepted). A grid is never
repeated within a test input; empty slots get `[[0]]`, and every task of the challenges file
gets an entry. `write_submission` validates with
`arc26.submit.validate` and writes with `arc26.submit.write_json_atomic`. `select_files(records,
challenges, out, nvarc=None)` does it all from files and reports how many slots each source
filled. Grouping is per test input rather than over all test inputs jointly, because each test
output is scored on its own and joint groups could repeat a grid in both slots.

## Answer forcing (run `P6.3-04`)

`Budget.forcing_kind` picks the forcing note: `"program"` (`FORCING_NOTE`), `"answer"`
(`ANSWER_FORCING_NOTE`: exactly one answer block per test input, nothing else), or `"answer2"`
(run `P6.3-05`: one or two answer blocks per test input, best first, with
`DIRECT_TWO_SYSTEM_PROMPT`). The reserve
counts the chosen note (`forcing_chars(kind)`). With `DIRECT_SYSTEM_PROMPT`, one round, and a
REPL that refuses code, the loop is the direct-answer arm: its answers are the record's
`direct_answers`.

## Tests

`tests/agent/test_tasks.py`, `test_engine.py`, `test_loop.py`, `test_select.py`, about 4 s on
the CPU:

- engine: request bodies (both effort transports, the thinking switch, sampling, seed),
  `reasoning_content` and `reasoning`, null content, malformed replies, the timing record with
  its task id, retries until a late server starts, exhausted retries, a dropped connection
  retried, HTTP 400 and 500 and read timeouts not retried, the pool's bound on concurrent
  requests, the bearer token, reasoning sent back with `preserve_thinking`, a second
  `asyncio.run` with the same engine, the default timeouts, `</think>` in content; against a
  stdlib HTTP server on localhost;
- loop with `ScriptedEngine`, an in-process fake REPL and a fake verifier: one-turn solve, a fix
  after feedback (the report's content), the round limit, the token budget lowering `max_tokens`,
  the context bound, an overlong reply, a reply without code, direct answers (valid and invalid),
  engine errors before and after a turn, the deadline during a call and before it, hostile
  programs rejected, sandbox attacks reported as cell errors, the cell cap, define-and-test in one
  block, the transcript, a REPL fault propagating, a verifier returning numpy values, a record
  with NaN that cannot be logged, the event loop staying responsive while the REPL and the
  verifier block, token sums over turns, the deadline between cells, `parse_answer` limits,
  import-only `try` blocks, `preserve_thinking`;
- forcing turns (revision 5): a cut reply's forcing request has thinking off, the cut reasoning
  carried, and the reserved cap, and its program verifies; every call names the effort, and the
  engine turns the forcing call's into the system note (no effort field, the template's text,
  none for `medium`); at most one forcing turn per cut reply and none after a cut forced reply;
  a forced reply without code goes on as usual; no forcing without `forcing_tokens`; the reserve
  in the attempt's tokens and in the context; a 60,000-character kept reasoning counted as its
  15,000 tokens; the record's counts round trip; a phase 2 opening with over 8,500 characters
  of earlier programs bounded by the count (its first turn as stated), a failed or odd count
  falling back to the character bound, the count's 64-token slack, an engine without a counter;
  `count_prompt_tokens` against a stdlib server (the request body; malformed replies, HTTP
  errors, no URL); a reply with thinking off whose text arrived as
  reasoning becomes the answer;
- runner: the concurrency limit (peak in-flight calls), task order, seeds, attempts per task,
  phase ids, the deadline skipping unstarted attempts, failing attempts and callbacks, the
  crash-safe log with a cut line, transcripts;
- selection hand cases: votes per attempt, ties by size, demotion and order, NVARC fill and
  promotion, direct answers after NVARC's first attempt, size before demotion, unverified
  programs by cell accuracy, fatal programs excluded, dedupe across sources, missing tasks,
  NVARC parsing and its `[[0]]` placeholders, numpy grids;
- end to end: `run_tasks` with `ScriptedEngine` on 10 synthetic tasks (one case each: correct at
  once, fix after feedback, hostile then correct, no code then correct, overlong then correct,
  direct answers only, engine error with NVARC's answer, define-and-test, always wrong, demoted)
  writes a `submission.json` that validates and scores 9 of 10.

## Limits and open questions

- `preserve_thinking` is off by default; Qwen's default (on) keeps the whole reasoning trace and
  gets exact prefix-cache hits but grows every prompt. Run `P6.3-02`'s revision 5 turns it on
  with forcing turns; in a 32k context an attempt then holds one long thinking turn, its forcing
  turn, and a short repair turn.
- The first prompt's count costs one `/tokenize` request per attempt (milliseconds); without it
  (an engine without `count_prompt_tokens`, a failed count) the first turn gives away about
  5,000 tokens of room to the character bound (revision 4's opening prompts: bound 7,598 to
  12,380, counted 2,494 to 7,291).
- `QWEN38_EFFORT_NOTES` copies the template's text; a template with other text costs the forcing
  turn its prefix-cache hit (and adds one sentence to its system message), nothing else. The
  spike's forcing check reports `prefix_kept`.
- An unverified program's output is a `"program"` candidate with 0 votes, since
  `CandidateSource` has no unverified kind.
- `StopReason` has no value for an infrastructure fault, so a REPL or verifier exception leaves
  no record.
- A record holding NaN or an infinity is not logged (an error is logged instead); it is still
  returned and passed to `on_record`.
- The verifier is called from up to `concurrency` threads at once.
