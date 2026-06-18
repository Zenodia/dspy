# NeMo Relay Integration — rung3 ACE ALFWorld

Instrumented entry point: `ace_relay_trace.py`  
Original (unmodified): `rung3_ace_alfworld.py`

---

## Quick Start

```bash
# smoke: 2 train + 1 eval
.venv/bin/python ace_relay_trace.py --smoke

# full: N_TRAIN train + N_TEST eval
.venv/bin/python ace_relay_trace.py --full
```

Traces land in:
```
traces/
  atif/   episode-<session_id>.json   # one structured trajectory per episode
  atof/   events.jsonl                # raw JSONL event stream, appended each run
```

---

## Plugins Enabled

### ATIF — Agent Trajectory Interchange Format
Captures one structured trajectory file per agent scope. Each episode becomes a
self-contained replay: scope hierarchy, mark events (goal, step, reflection,
playbook), and LLM call spans (start/end timing, model, result summary).

```python
# Created per episode — unique session_id/trajectory_id per file
atif = nemo_relay.AtifExporter(
    str(uuid.uuid4()),
    "rung3-ace-alfworld",
    "1.0",
    model_name=REASONER_MODEL,
)
atif.register(f"atif-ep-{label}")   # register before scope opens
# ... episode runs ...
with open(f"traces/atif/episode-{label}.json", "w") as f:
    f.write(atif.export_json())     # export after scope closes
atif.deregister(f"atif-ep-{label}")
```

### ATOF — Agent Trajectory Observability Format
Appends every lifecycle event (scope open/close, mark events, LLM call start/end)
to a single JSONL file. Use for dashboards, post-run grep, and timeline debugging.

```python
# Registered once globally; shutdown at end of main()
cfg = nemo_relay.AtofExporterConfig()
cfg.output_directory = "./traces/atof"
cfg.filename = "events.jsonl"
cfg.mode = nemo_relay.AtofExporterMode.Append   # accumulates across runs
atof = nemo_relay.AtofExporter(cfg)
atof.register("atof-exporter")
# ... run ...
atof.force_flush()
atof.shutdown()
```

---

## Async Compatibility

The full rung2/rung3 stack is **synchronous** (Haystack `NvidiaChatGenerator`,
DSPy `Predict`, `_throttle` loop). `nemo_relay.llm.execute()` is **async-only**
with no documented sync equivalent. `nemo_relay.tools.call/call_end` is sync.

### Solution used: `nemo_relay.llm.call()` / `call_end()` — sync

`nemo_relay.llm.call()` and `call_end()` are sync and produce proper ATIF model
steps. Despite docs only showing `execute()` (async), the manual lifecycle API
is sync and fully functional in v0.4.0.

**For retry loops** (`_reason_traced`, reflector block): call `llm.call()` /
`call_end()` directly per attempt — each attempt gets its own ATIF model step.

**For single-shot calls** (DSPy `self.update()`): use `_LLMSpan`, a thin sync
context manager that wraps `llm.call()` / `call_end()`:

```python
with _LLMSpan("memory-updater", SUMMARIZER_MODEL, content, ep_handle) as span:
    upd = self.update(...)
    span.end(result={"new_fact": upd.new_fact, "progress": upd.progress})
```

`nemo_relay.llm.execute()` (async) is NOT needed — the sync manual lifecycle
API gives full ATIF model-step coverage.

---

## Scope Hierarchy

```
ace-train  (ScopeType.Agent)
└── episode-train1  (ScopeType.Agent)
│       events: cache_lookup, playbook_injected, episode_goal
│       events: agent_step × N
│       events: episode_outcome
│       events: llm_call_start/end × N  (memory-updater, reasoner)
│       events: reflection_summary, playbook_updated
│       events: llm_call_start/end  (reflector)
└── episode-train2 ...

ace-eval  (ScopeType.Agent)
└── episode-eval-0  (ScopeType.Agent)
│       events: cache_lookup, playbook_injected, episode_goal
│       events: agent_step × N
│       events: episode_outcome
│       events: llm_call_start/end × N
└── ...
    events: eval_complete
```

---

## Traced Events Reference

### Episode scope events

| Event | Fired by | Key fields |
|---|---|---|
| `cache_lookup` | `run_episode_traced`, eval loop | `cache_key`, `hit`, `prefill_facts_count`, `prefill_dead_ends_count` |
| `playbook_injected` | same | `bullet_count`, `playbook` (rendered text) |
| `episode_goal` | `TracedMemoryAgent.forward` | `goal`, `common_sense_hint`, `prefill_facts`, `prefill_dead_ends` |
| `agent_step` | `forward` — after every `env.step()` | `step`, `observation`, `action_taken`, `progress`, `plan`, `recalled_facts`, `dead_ends`, `result_obs`, `is_done` |
| `episode_outcome` | `run_episode_traced` | `label`, `goal`, `outcome`, `steps`, `success`, `trajectory_tail` |

### LLM lifecycle events (via `_LLMSpan`)

| Event | Provider | Fired when |
|---|---|---|
| `llm_call_start` | `memory-updater` | Before every DSPy `self.update()` call |
| `llm_call_end` | `memory-updater` | After; includes `new_fact`, `dead_end`, `progress`, `plan` |
| `llm_call_start` | `reasoner` | Before every Haystack `NvidiaChatGenerator.run()` attempt |
| `llm_call_end` | `reasoner` | After; includes `action_chosen`, `attempt`, `duration_ms` |
| `llm_call_start` | `reflector` | Before every DSPy `Predict(Reflect)` attempt |
| `llm_call_end` | `reflector` | After; includes `helpful_ids`, `harmful_ids`, `new_strategies` |

### RPM / failure events

| Event | Key fields | Notes |
|---|---|---|
| `llm_rate_limit_hit` | `provider`, `model`, `step`, `attempt`, `wait_seconds`, `error` | Fired on 429 / "rate limit" / "too many requests" match before each retry wait |
| `reasoner_fallback` | `step`, `fallback_action`, `last_error` | Fired when all `_RPM_MAX_RETRIES` attempts exhaust; heuristic action chosen |

### Reflection events

| Event | Key fields |
|---|---|
| `reflection_summary` | `goal`, `outcome`, `steps_taken`, `last_observation`, `helpful_strategy_ids`, `harmful_strategy_ids`, `next_best_strategies`, `playbook_size_before`, `current_playbook` |
| `playbook_updated` | `playbook_size_after`, `bullets_delta`, `playbook_snapshot` |

### Phase events

| Event | Scope | Key fields |
|---|---|---|
| `phase_start` | `ace-train` | `phase`, `n_episodes`, `reasoner_model`, `summarizer_model` |
| `phase_end` | `ace-train` | `phase`, `playbook_size`, `playbook` |
| `phase_start` | `ace-eval` | `phase`, `n_episodes`, `frozen_playbook_size`, `frozen_playbook` |
| `eval_complete` | `ace-eval` | `success_pct`, `avg_return`, `avg_steps`, `n_test`, `playbook_final` |

---

## RPM Retry Logic

All three LLM call sites (reasoner, memory-updater via DSPy, reflector) share
the same retry parameters:

```python
_RPM_WAIT_BASE    = 15.0   # seconds — conservative floor under 40 RPM / 60s window
_RPM_MAX_RETRIES  = 5
_RPM_WAIT_CAP     = 60.0   # max single wait; doubles each attempt up to this cap
```

Detection matches any of: `"429"`, `"rate limit"`, `"too many requests"`, `"rpm"`
(case-insensitive) in the exception string.

Behavior per attempt:
1. Fire `llm_rate_limit_hit` event (captured in ATOF)
2. Print yellow warning with wait time
3. `time.sleep(wait)`
4. `wait = min(wait * 2, _RPM_WAIT_CAP)`
5. Retry

On exhaustion: reasoner falls back to heuristic action + fires `reasoner_fallback`
event. Reflector raises (episode cache is already saved; playbook update skipped).

This is **reactive** (catches actual failures). The existing `_throttle()` from
rung2 is **proactive** (slides a 28-RPM window before each call). Both run
together for defense-in-depth.

---

## Architecture Notes

### `TracedMemoryAgent` (subclass of `StructuredMemoryAgent`)

Overrides `forward()`. Behaviorally identical to rung2 — same memory logic,
same action filtering, same grounding. Additions:

- Accepts `self._ep_handle` (set by the episode wrapper before `forward()`,
  cleared after) to route events to the correct scope.
- Replaces `_reason()` calls with `_reason_traced()`.
- Wraps `self.update()` in `_LLMSpan("memory-updater", ...)`.
- Emits `episode_goal` once and `agent_step` after every `env.step()`.

### `_reason_traced()`

Drop-in for `rung2._reason()`. Rebuilds the same user message, calls
`_reasoner().run()` through the same Haystack path, extracts action the same
way. Differences: `_LLMSpan` per attempt, RPM retry loop, `reasoner_fallback`
on exhaustion.

### `run_episode_traced()`

Mirrors `rung3.run_episode()` exactly. Wraps the whole episode in
`nemo_relay.scope.scope(f"episode-{label}", ScopeType.Agent)`. Moves the
reflector's retry handling inline (was a single call with `num_retries=10` on
the LM object; now explicit loop with per-attempt events).

---

## Output Files

| File | Format | Content |
|---|---|---|
| `traces/atif/episode-<id>.json` | ATIF v1.7 JSON | One per agent scope; structured trajectory with scope tree, mark steps, LLM spans |
| `traces/atof/events.jsonl` | NDJSON | One line per lifecycle event; append mode accumulates across runs |
| `results_rung3_relay.json` | JSON | Identical schema to `results_rung3.json`; metrics + per-game records |
