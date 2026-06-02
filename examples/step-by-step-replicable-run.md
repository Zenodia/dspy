# Baseline vs MIPROv2 vs GEPA on a single ALFWorld game — Replicable Run

A small, runnable comparison of an un-optimized agent vs two DSPy prompt
optimizers (**MIPROv2**, **GEPA**) on **one** ALFWorld text-game, measuring
**how fast each solves it** (steps to success; unsolved within the budget = a
loss). Models come from [build.nvidia.com](https://build.nvidia.com), served
OpenAI-compatibly.

Built on the DSPy *Finetuning Agents* tutorial
([dspy.ai/tutorials/games](https://dspy.ai/tutorials/games/)) and the local
`textgame_exp.ipynb` ALFWorld ReAct agent.

Script: `mipro_speed_alfworld.py`

> **Origin / why this shape.** This started from the multiple-choice QA notebook
> (`CustomDatasetmultiChoicesQA.ipynb`). That task gave no headroom (the NVIDIA
> reasoning model scored ~90–100%), so we moved to ALFWorld, a long-horizon
> agent benchmark. Two practical findings shaped the final setup:
> 1. **Reasoning model is too slow here.** `nemotron-49b` emits a long hidden
>    think-trace every step (~31s/step), so one 40-step rollout ≈ 21 min and a
>    full optimizer compile would take many hours. We switched the agent to the
>    non-reasoning **`meta/llama-3.3-70b-instruct`** (~3–13s/step) — the same
>    model `textgame_exp.ipynb` uses.
> 2. **Optimizer compiles are rollout-bound.** Each MIPRO/GEPA "metric call" is a
>    full game rollout (up to MAX_ITERS LM calls). To finish in ~1 hr on a single
>    game we cap MAX_ITERS and give the optimizers a tiny budget (this is a
>    *smoke*, not a full optimization).

---

## 0. Models

| Role | Model |
|------|-------|
| Agent (baseline / MIPRO / GEPA) | `meta/llama-3.3-70b-instruct` |
| MIPRO prompt model + GEPA reflection | `meta/llama-3.3-70b-instruct` (same model) |

Served via `https://integrate.api.nvidia.com/v1`, litellm `openai/` prefix, key
from `.env` (`NVIDIA_API_KEY`).

> **Rate limit:** build.nvidia.com free tier is **40 RPM**. The script runs
> **serial** (`NUM_THREADS=1`) with `num_retries=10` (litellm exponential
> backoff) to ride out `429`s.

## 1. `.env`

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NeMoTronModel=nvidia/llama-3.3-nemotron-super-49b-v1.5   # used only if you switch the agent back to reasoning
```

## 2. Environment + dependencies

```bash
cd /home/ubuntu/dspy
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e . python-dotenv optuna
# ALFWorld stack (already present on this box): alfworld==0.3.5, textworld==1.7.0, multiprocess
#   if missing: .venv/bin/pip install "alfworld==0.3.5" && .venv/bin/alfworld-download
```

Game data lives under `~/.cache/alfworld` (here: `/ephemeral/cache/alfworld/json_2.1.1`);
`dspy.datasets.alfworld.AlfWorld` loads it automatically.

Verify: `.venv/bin/python -c "import dspy, alfworld, textworld; from dspy.teleprompt import MIPROv2, GEPA; print(dspy.__version__)"` → `3.3.0b1`

## 3. The `spawn` gotcha

ALFWorld's env pool uses `multiprocess` **`spawn`**, which re-imports the module
in every worker. So `AlfWorld()` is created **only inside `main()`** (the global
`alfworld` stays `None` on import). Always run the script as a **file**, never
via stdin/heredoc.

## 4. The agent (simple windowed ReAct)

```python
class Agent(dspy.Module):
    def __init__(self, max_iters=12, window=8):
        # planning seed instruction; MIPRO/GEPA mutate THIS
        self.react = dspy.Predict(dspy.Signature(
            "task, trajectory, possible_actions: list[str] -> action", INSTRUCTIONS))
    def forward(self, idx):
        # loop up to max_iters:
        #   window = last `window` (action, obs) turns   <-- bounds the prompt
        #   pred   = react(task, window, admissible_commands)
        #   action = clean_action(pred.action, admissible)  # strip CoT/delimiter junk, snap to a valid command
        #   step env; log turn to disk; stop if done
        # unsolved within max_iters (or any context/API error) => success=0 (LOSS)
```

Key choices you asked for:
- **Sliding window** of the last `WINDOW=8` turns instead of a separate "agent
  memory" add-on — it carries recent game state and keeps the prompt small.
- **Loss on budget:** if the agent doesn't finish within `MAX_ITERS=12` steps
  (or the context/API errors mid-episode), it's a loss. The window means the
  context never actually overflows.
- **`clean_action`**: models leak chain-of-thought / DSPy field delimiters into
  the action; we strip that and snap to a valid admissible command. (Without it
  the agent emits `go to desk 1 [[ ## completed ## ]]` → env says "Nothing
  happens" → never progresses.)

### MIPRO vs GEPA difference

```python
# MIPROv2: scalar "solve fast" metric (1.0 if solved in 1 step -> 0; 0 if unsolved)
def metric_speed(ex, pred, trace=None):
    return 0.0 if not pred.solved else max(0.0, 1 - (pred.steps - 1) / MAX_ITERS)

# GEPA: same score PLUS text feedback (the failed trajectory) for the reflection LM
def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    fb = "SUCCESS..." if pred.solved else "FAILURE: looped/unsystematic...\n<trajectory tail>"
    return dspy.Prediction(score=metric_speed(gold, pred), feedback=fb)
```

MIPRO sees only the number; GEPA also reads the failure trajectory and uses the
reflection LM to rewrite the instruction.

### On-disk memory reflects train vs eval

Every rollout writes a grep-friendly, phase-tagged log to `alfworld_memory/`
(format inspired by [github.com/Zenodia/memory_ondisk](https://github.com/Zenodia/memory_ondisk)):

```
baseline__game1650__ep0001.txt     # baseline eval
train__game3020__ep0003.txt        # a MIPRO/GEPA compile rollout
eval__game1650__ep0007.txt         # final optimized run on the target game
```

Each file header carries `@PHASE:{baseline|train|eval}@` and `@EPISODE:NNNN@`, so
the **training rollouts** (optimizer search) and **eval runs** (the reported
numbers) are separable on disk. Grep examples:

```bash
grep -l '@PHASE:train@' alfworld_memory/*.txt   # all optimizer training rollouts
grep -h '@TASK:'        alfworld_memory/eval__*  # the eval task(s)
```

## 5. Run

```bash
.venv/bin/python mipro_speed_alfworld.py --idx 1650                 # full 3-way smoke
.venv/bin/python mipro_speed_alfworld.py --idx 1650 --baseline-only # just the baseline
```

Knobs (top of script): `MAX_ITERS=12`, `WINDOW=8`, `GEPA_METRIC_CALLS=5`, MIPRO
`num_candidates=2 / num_trials=2`, train=2 / val=1 game. Artifacts:
`optimized_mipro_speed.json`, `optimized_gepa_speed.json`.

---

## 6. Results

Reproduced 2026-06-02. Single game **idx 1650** — task *"put some cd on shelf"*.
Same model (`llama-3.3-70b-instruct`) for all three.

```
========================================================
  ALFWorld single-game SPEED: baseline vs MIPROv2 vs GEPA (idx=1650)
========================================================
              solved   steps  seconds
  baseline     False      12       30
  MIPROv2      False      12        4
  GEPA         False      12        4
========================================================
```

| Agent | Solved? | Steps to solve | Wall-clock |
|-------|---------|----------------|-----------|
| baseline | **No** (loss) | 12 (hit budget) | 30s |
| MIPROv2  | **No** (loss) | 12 (hit budget) | 4s* |
| GEPA     | **No** (loss) | 12 (hit budget) | 4s* |

\* near-instant because the optimized agents took the same early actions as the
baseline, so litellm served them from its disk cache.

### Reading the result honestly

- **The pipeline works end-to-end** for all three: a 70b agent drives ALFWorld
  TextWorld rollouts, valid actions are issued (`clean_action`), the sliding
  window keeps the prompt bounded, the loss-on-budget rule fires, and MIPRO &
  GEPA both compile and run on the single game. That was the goal of the smoke.
- **All three lost** on this game. Two reasons, both expected:
  1. **`MAX_ITERS=12` is a very tight budget** for "put some cd on shelf" — the
     agent must locate the cd among many drawers/desks/shelves first. With only
     the last 8 turns visible it re-checks places and runs out of steps. (At
     `MAX_ITERS=40` the same baseline still lost but explored further — the
     model's *strategy*, not the budget, is the main limit.)
  2. **The optimizer budget is a smoke, not a real optimization.** MIPRO ran
     2 candidates × 2 trials and GEPA 5 metric calls on 1–2 train games. That is
     far too little signal to discover a materially better instruction —
     especially when **no training rollout succeeded**, so neither optimizer had
     a positive example to learn from (GEPA's base valset score was 0/1).
- **So MIPRO ≈ GEPA ≈ baseline here** — not because the optimizers are
  equivalent, but because there was neither enough step budget to succeed nor
  enough optimization budget to improve. This is the honest outcome of a
  time-boxed (<1 hr, 40 RPM) single-game smoke.

### What would actually show a MIPRO-vs-GEPA gap

- Raise `MAX_ITERS` (e.g. 40–50) so games are winnable, and pick easier task
  types (single-object `pick_and_place`) so some training rollouts succeed.
- Give the optimizers a real budget (MIPRO `auto="light"`, GEPA `auto="light"`
  or `max_metric_calls` in the hundreds) over **tens** of training games.
- Score on a **held-out set of games**, not one — single-game numbers are noisy.
- Budget the time: at ~13s/step and 40 RPM this is **hours**, so run it in the
  background rather than interactively. On sparse-reward agent tasks, GEPA
  typically pulls ahead of MIPRO once it has at least one successful trajectory
  to reflect on.

### Notes / known-benign
- At interpreter shutdown you may see `Exception ignored in: AlfWorld.__del__ …
  can't create new thread at interpreter shutdown`. Harmless pool teardown.
