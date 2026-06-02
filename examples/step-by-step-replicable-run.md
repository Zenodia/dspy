# MIPROv2 vs GEPA on Multiple-Choice QA — Replicable Run

Compares two DSPy prompt optimizers — **MIPROv2** and **GEPA** — on a
multiple-choice QA task, using an NVIDIA reasoning model served
OpenAI-compatibly from [build.nvidia.com](https://build.nvidia.com).

Adapted from the `CustomDatasetmultiChoicesQA.ipynb` notebook. The notebook's
private `/workspace/nvdata/train.csv` is replaced by a small inline trap-style
MCQ dataset so the script is fully self-contained, and `KNNFewShot` is swapped
for a MIPROv2-vs-GEPA comparison.

---

## 0. Prerequisites

- Python 3.12
- An NVIDIA API key (`nvapi-...`) from <https://build.nvidia.com> with access to
  `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- This repo checked out at `/home/ubuntu/dspy`.

## 1. `.env`

Create `/home/ubuntu/dspy/.env`:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NeMoTronModel=nvidia/llama-3.3-nemotron-super-49b-v1.5
```

The model is OpenAI-compatible; DSPy reaches it through
`https://integrate.api.nvidia.com/v1` with the `openai/` litellm prefix.

## 2. Environment + dependencies

The system Python is PEP-668 "externally managed", so use a venv.

```bash
cd /home/ubuntu/dspy
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .          # installs dspy (editable) + litellm + openai
.venv/bin/pip install python-dotenv # .env loader
.venv/bin/pip install optuna        # REQUIRED by MIPROv2 (Bayesian search backend)
```

Verify:

```bash
.venv/bin/python -c "import dspy; from dspy.teleprompt import MIPROv2, GEPA; print(dspy.__version__)"
# -> 3.3.0b1
```

> Note: activate with `source .venv/bin/activate` if you prefer; the commands
> above call `.venv/bin/python` directly so activation is optional.

## 3. Smoke-test the LM connection

```bash
.venv/bin/python - <<'PY'
import os, dspy
from dotenv import load_dotenv
load_dotenv()
lm = dspy.LM("openai/"+os.environ["NeMoTronModel"],
             api_key=os.environ["NVIDIA_API_KEY"],
             api_base="https://integrate.api.nvidia.com/v1",
             temperature=0.6, max_tokens=4000)
print(lm("Reply with only the letter B."))
PY
```

A reasoning model returns a dict with `text` **and** `reasoning_content`.
The script's metric parses the first `A–D` letter out of `text`, so the
chatty think-trace doesn't matter.

## 4. Run the comparison

```bash
.venv/bin/python mipro_vs_gepa_mcq.py
```

Runs three stages and prints a final table:

1. **Baseline** — un-optimized `ChainOfThought`.
2. **MIPROv2** — `auto="light"`, bool metric (`metric_simple`). Bootstraps
   few-shot demos + Bayesian-searches instruction candidates.
3. **GEPA** — `auto="light"`, feedback metric (`metric_feedback` returns
   `dspy.Prediction(score, feedback)`); a `reflection_lm` reads the feedback
   text and *rewrites* the prompt.

Artifacts: `optimized_mipro.json`, `optimized_gepa.json` (the optimized
programs — reload with `prog.load(path)`).

### Timing note

Each LM call to the 49B reasoning model is ~15–25 s (it spends tokens on the
think trace). A 10-item eval ≈ 3–4 min cold; optimizer compiles make many
calls. DSPy's litellm disk cache makes repeat calls near-instant, so re-runs
are much faster than the first.

---

## 5. Key code (what differs between the two optimizers)

```python
# --- shared task ---
class BasicQA(dspy.Signature):
    """Answer the multiple-choice question. Reply with the single letter
       (A, B, C, or D) of the correct option."""
    question = dspy.InputField()
    choices  = dspy.InputField()
    answer   = dspy.OutputField(desc="the single letter of the correct choice")

program = dspy.ChainOfThought(BasicQA)

# --- MIPROv2: plain bool metric ---
def metric_simple(example, pred, trace=None):
    return _letter(pred.answer) == example.answer.strip().upper()

mipro = MIPROv2(metric=metric_simple, auto="light", num_threads=8)
mipro_prog = mipro.compile(program, trainset=trainset, valset=valset,
                           requires_permission_to_run=False)

# --- GEPA: feedback metric drives reflective rewriting ---
def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = _letter(pred.answer) == gold.answer.strip().upper()
    fb = "Correct." if correct else f"Incorrect; correct option is {gold.answer}. ..."
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=fb)

gepa = GEPA(metric=metric_feedback, auto="light",
            reflection_lm=reflection_lm, num_threads=8, track_stats=True)
gepa_prog = gepa.compile(program, trainset=trainset, valset=valset)
```

**The core distinction:** MIPROv2 only sees a *number* per example; GEPA also
sees *text feedback* and uses a reflection LM to reason about failures and
rewrite the instruction. The richer the feedback string, the more GEPA can do.

---

## 6. Results

<!-- RESULTS_PLACEHOLDER -->
_Run in progress — table is filled in below once MIPROv2 and GEPA compiles
finish._

Dataset: 40 trap-style MCQs (20 train / 10 val / 10 test).

| Optimizer | Test accuracy |
|-----------|---------------|
| baseline  | 90.0%         |
| MIPROv2   | _pending_     |
| GEPA      | _pending_     |
