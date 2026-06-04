# GEPA + Haystack memory agent on ALFWorld — step-by-step

A hybrid agent for the ALFWorld text game. **Haystack** is the specialized
agentic layer (NVIDIA-served memory + a tool-calling reasoner); **DSPy/GEPA**
optimizes the one prompt that matters. No NeMo Agent Toolkit — just Haystack +
the NVIDIA integration.

File: `gepa_haystack_alfworld.py`

---

## 0. What this is

| Layer | Does | Who owns it |
|---|---|---|
| Summarizer | `(task, recent_obs, history) -> memory_item` | **DSPy `Predict`** — the only GEPA-optimized prompt |
| Memory store | embed + write memory items, recall top-k relevant | **Haystack** `NvidiaDocumentEmbedder` / `NvidiaTextEmbedder` / `InMemoryEmbeddingRetriever` |
| Reasoner | read task + recalled memory + admissible actions -> pick next action | **Haystack** `NvidiaChatGenerator` (kimi-k2.6) + per-step `Tool` |

Experimental question: *can GEPA learn to summarize observations into memory
items that let a FIXED reasoner finish in fewer steps?*

---

## 1. The objective — why "fewest steps" is the metric (Q-learning grounded)

ALFWorld is an episodic MDP with a sparse terminal reward `r_T = 1` on success.
The Q-learning / discounted-return target

```
Q(s,a) = r + gamma * max_a' Q(s',a')        # Bellman, gamma in (0,1)
G_0     = sum_t gamma^t * r_t  =  gamma^T * 1[success]
```

collapses (sparse terminal reward) to `gamma^T`. Since `gamma < 1`, `gamma^T` is
**strictly decreasing in T**, so maximizing `E[G_0]` *is* reaching the goal in
the fewest steps — no hand-tuned step penalty.

Pick `gamma` so the discount horizon equals the step budget `H = 50`
(return decays to exactly `1/e` at the limit):

```
gamma = e^(-1/(H-1)) ~ 0.9798
gamma^0   = 1.000   # solved in 1 step (perfect, achievable)
gamma^49  = 1/e ~ 0.368   # slowest possible success
```

That's the scalar GEPA maximizes (`metric_feedback` returns it as `score`).

---

## 2. Prerequisites

- The DSPy repo venv at `/home/ubuntu/dspy/.venv` (Python 3.12).
- `.env` at repo root with `NVIDIA_API_KEY=nvapi-...` (build.nvidia.com key).
- ALFWorld game data (DSPy's `dspy.datasets.alfworld.AlfWorld` downloads on first use).

```bash
.venv/bin/pip install "haystack-ai>=2.18.1,<3.0.0" "nvidia-haystack~=0.3.0"
```

No NeMo Agent Toolkit. Version pins copied from NVIDIA's Haystack example, minus
the toolkit. Installed at build time: `haystack-ai 2.30.0`, `nvidia-haystack 0.3.0`.

---

## 3. Models (all via `https://integrate.api.nvidia.com/v1`)

| Role | Model | Notes |
|---|---|---|
| Reasoner + GEPA reflection | `moonshotai/kimi-k2.6` | tool calling + `thinking` mode |
| Summarizer (GEPA target) | `nvidia/llama-3.3-nemotron-super-49b-v1` | swap to `...v1.5` (`.env` `NeMoTronModel`) if v1 unavailable |
| Embedder | `nvidia/nv-embedqa-e5-v5` | retrieval-tuned, 1024-d |

**One shared 40 RPM pool.** All three hit the same endpoint, so rate limit is
global. kimi adds tokens + latency (16k-capable `thinking`), **not** more calls.

---

## 4. The per-step loop (`HaystackMemoryAgent.forward`)

For each of up to `MAX_ITERS = 50` steps:

1. **Summarize** — DSPy `Predict` turns `(task, recent_obs, last-6 history)` into
   one `memory_item`. *(GEPA-optimized step.)*
2. **Store** — `NvidiaDocumentEmbedder` embeds the new memory item; written to a
   per-episode `InMemoryDocumentStore` (`DuplicatePolicy.OVERWRITE` — identical
   summaries collapse instead of crashing).
3. **Recall** — `NvidiaTextEmbedder` embeds `task + recent_obs`;
   `InMemoryEmbeddingRetriever` pulls top-`k=3` relevant memories.
4. **Reason** — `NvidiaChatGenerator` (kimi) gets the task, recalled memory, and
   admissible actions, with **one `Tool` rebuilt each step whose `action` enum ==
   the current admissible commands**. `tool_choice="required"` -> the returned
   action is guaranteed legal (zero parse failures, zero illegal moves).
5. **Step** the env; append observation; stop on `done`.

Per step ~= 4 NVIDIA calls (summarize + doc-embed + query-embed + reason). Only
the new memory item is embedded each step (old ones are not re-embedded).

### Two gotchas baked into the code
- **kimi `thinking`** is a non-OpenAI field — it must ride in
  `generation_kwargs={"extra_body": {"chat_template_kwargs": {"thinking": True}}}`,
  not as a top-level kwarg (the OpenAI client rejects unknown kwargs).
- **Spawn safety** — ALFWorld's pool re-imports this module in every worker, so
  `alfworld` stays `None` at import and Haystack components are lazy singletons
  (`_HAYSTACK` cache) built only in the main process on first `forward`.

---

## 5. Rate-limit handling (40 RPM)

- `NUM_THREADS = 1` (serial).
- `_throttle()` enforces `>= 1.8 s` between Haystack NVIDIA calls (~33/min).
- `num_retries = 10` on the DSPy LMs -> litellm exponential backoff on any 429
  that slips through (DSPy calls aren't counted by `_throttle`).

Cost of the step budget: a worst-case 50-step rollout ~= 200 calls ~= 5 min at
40 RPM. The **full GEPA run is multi-hour** at these limits — expect to leave it.

---

## 6. Run it

Smoke test the pipeline first (1 task, prints each action):

```bash
.venv/bin/python gepa_haystack_alfworld.py --smoke 1
```

Full baseline-vs-GEPA comparison:

```bash
.venv/bin/python gepa_haystack_alfworld.py
```

Knobs (top of file): `N_TRAIN=6`, `N_VAL=6`, `N_TEST=10`, `GEPA_METRIC_CALLS=30`.
Raise for a stronger result, but multiply the wall-clock by the 40 RPM ceiling.

---

## 7. What runs, what it prints

1. **Baseline** — un-optimized summarizer, evaluated on `N_TEST` tasks.
2. **GEPA** — `metric_feedback` (discounted return + textual diagnosis) drives
   the kimi reflection LM to rewrite the summarizer's instruction over
   `GEPA_METRIC_CALLS` rollouts. Best program saved to
   `optimized_gepa_haystack_alfworld.json`.
3. **Comparison table** — for baseline and GEPA: avg **discounted return** (the
   objective), raw **success %**, and avg **steps-to-solve**.

```
============================================================
  GEPA + Haystack memory agent on ALFWorld
  objective = discounted return  gamma = e^(-1/49) ~ 0.9798
============================================================
                 return    success%   avg_steps
  baseline        0.xxx       xx.x%        xx.x
  gepa            0.xxx       xx.x%        xx.x
============================================================
```

---

## 8. Extending

- **Reranker**: add `NvidiaRanker` after retrieval for sharper memory recall —
  costs +1 call/step against the 40 RPM budget (left off by default).
- **Optimize the reasoner too**: would require bridging kimi's prompt into a
  DSPy predictor so GEPA can mutate it (the rejected "Arch B"); current design
  keeps the reasoner fixed and optimizes only the summarizer.
- **Different objective**: swap `discounted_return` for tiered/linear shaping —
  but discounting is the theoretically grounded choice for fewest-steps.
