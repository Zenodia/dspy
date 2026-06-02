"""
MIPROv2 vs GEPA prompt-optimizer comparison on a multiple-choice QA task.

Adapted from the CustomDatasetmultiChoicesQA notebook
(github.com/Zenodia/dspy ... examples/notebooks/CustomDatasetmultiChoicesQA.ipynb).

Differences vs the notebook:
  * The notebook's private /workspace/nvdata/train.csv is replaced by a small
    inline multiple-choice dataset so the script is self-contained.
  * The notebook used mixtral-8x22b + KNNFewShot. Here the task LM is the
    NVIDIA reasoning model from .env (llama-3.3-nemotron-super-49b-v1.5),
    served OpenAI-compatibly through https://integrate.api.nvidia.com/v1,
    and we compare two prompt optimizers: MIPROv2 and GEPA.

Run:
    .venv/bin/python mipro_vs_gepa_mcq.py
"""

import os
import time

import dspy
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# 1. LM setup  (build.nvidia.com, OpenAI-compatible scope)                     #
# --------------------------------------------------------------------------- #
load_dotenv()

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
TASK_MODEL = os.environ.get("NeMoTronModel", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# dspy routes through litellm; the `openai/` prefix tells litellm to use the
# OpenAI chat protocol against the custom api_base below.
task_lm = dspy.LM(
    f"openai/{TASK_MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=0.6,
    max_tokens=8000,  # reasoning models spend tokens on the think trace
)
dspy.configure(lm=task_lm)

# GEPA needs a (typically strong) model to *reflect* on failures and rewrite
# prompts. We reuse the same reasoning model at high temperature.
reflection_lm = dspy.LM(
    f"openai/{TASK_MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=1.0,
    max_tokens=16000,
)

# --------------------------------------------------------------------------- #
# 2. Inline multiple-choice dataset                                           #
# --------------------------------------------------------------------------- #
# Each item: (question, [4 choices], correct_letter)
# Trap-style MCQ: negation, careful-reading, counting, lateral/CRT logic, and
# commonly-confused facts. A strong reasoning model still slips on these, which
# leaves headroom for the optimizers to improve the prompt.
RAW = [
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
     ["$0.10", "$0.05", "$1.00", "$0.15"], "B"),
    ("How many times does the letter 'a' appear in the word 'banana'?", ["1", "2", "3", "4"], "C"),
    ("Which of these is NOT a prime number?", ["2", "3", "9", "11"], "C"),
    ("If you are running a race and you overtake the person in 2nd place, what place are you in now?",
     ["1st", "2nd", "3rd", "Last"], "B"),
    ("A pound of feathers and a pound of bricks: which weighs more?",
     ["The bricks", "The feathers", "They weigh the same", "Cannot be determined"], "C"),
    ("Which word is spelled correctly?", ["Neccessary", "Necesary", "Necessary", "Neccesary"], "C"),
    ("All Bloops are Razzies and all Razzies are Lazzies. Are all Bloops definitely Lazzies?",
     ["No", "Only some", "Yes", "Cannot tell"], "C"),
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", ["8", "9", "17", "0"], "B"),
    ("Which of the following is the largest?", ["0.9", "0.099", "0.1", "0.19"], "A"),
    ("Mary's father has five daughters: Nana, Nene, Nini, Nono, and ___. What is the fifth daughter's name?",
     ["Nunu", "Mary", "Nana", "None of these"], "B"),
    ("How many months have 28 days?", ["1", "2", "11", "12"], "D"),
    ("Which is heavier: 1 kilogram or 1000 grams?", ["1 kilogram", "1000 grams", "They are equal", "Depends on material"], "C"),
    ("What is the next number in the sequence: 2, 4, 8, 16, ...?", ["20", "24", "32", "64"], "C"),
    ("Which sentence uses 'their' correctly?",
     ["Their going home.", "The dogs wagged their tails.", "Its over their.", "Their is a problem."], "B"),
    ("A clock shows 3:15. What is the angle between the hour and minute hands?",
     ["0 degrees", "7.5 degrees", "37.5 degrees", "90 degrees"], "B"),
    ("Which of these is NOT a mammal?", ["Whale", "Bat", "Platypus", "Penguin"], "D"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
     ["100 minutes", "5 minutes", "20 minutes", "1 minute"], "B"),
    ("Which number is both a perfect square and a perfect cube?", ["8", "16", "36", "64"], "D"),
    ("'I am lying.' If this statement is spoken, it is best described as a:",
     ["True statement", "False statement", "Paradox", "Question"], "C"),
    ("Which planet has the most moons (as of 2024)?", ["Jupiter", "Saturn", "Uranus", "Neptune"], "B"),
    ("What is 0 divided by 5?", ["0", "5", "Undefined", "Infinity"], "A"),
    ("Which of these is NOT one of the primary colors of light (RGB)?",
     ["Red", "Green", "Yellow", "Blue"], "C"),
    ("A car travels 60 miles in 1 hour, then 60 miles in 2 hours. What is its average speed?",
     ["60 mph", "45 mph", "40 mph", "30 mph"], "C"),
    ("How many degrees are in the interior angles of a triangle, total?", ["90", "180", "270", "360"], "B"),
    ("Which is the odd one out?", ["Square", "Circle", "Triangle", "Rectangle"], "B"),
    ("If today is Monday, what day will it be 100 days from now?",
     ["Monday", "Tuesday", "Wednesday", "Thursday"], "C"),
    ("Which of these words is a palindrome?", ["Level", "World", "House", "Table"], "A"),
    ("What is the smallest two-digit prime number?", ["10", "11", "13", "17"], "B"),
    ("Forest is to trees as library is to ___?", ["Books", "Reading", "Quiet", "Building"], "A"),
    ("Which of these is NOT a noble gas?", ["Helium", "Neon", "Nitrogen", "Argon"], "C"),
    ("A rope ladder hangs over the side of a ship. Rungs are 1 ft apart; tide rises 1 ft/hr. After 3 hours, how many rungs are underwater if 2 were underwater at the start?",
     ["2", "3", "5", "8"], "A"),
    ("Which fraction is the largest?", ["3/4", "5/8", "7/10", "2/3"], "A"),
    ("How many sides does a 'nonagon' have?", ["7", "8", "9", "10"], "C"),
    ("Which statement about the equator is FALSE?",
     ["It divides Earth into hemispheres", "It is the longest line of latitude", "It passes through Brazil", "It is colder than the poles"], "D"),
    ("What comes next: J, F, M, A, M, ...?", ["J", "A", "N", "S"], "A"),
    ("If 5 + 3 = 28, 9 + 1 = 810, then 8 + 6 = ?", ["214", "148", "1428", "68"], "A"),
    ("Which of these is NOT a programming language?", ["Python", "Java", "Cobra", "HTML"], "D"),
    ("A snail climbs 3 ft up a 10 ft well by day but slips 2 ft each night. How many days to get out?",
     ["10", "8", "7", "5"], "B"),
    ("Which weighs the same as 16 ounces?", ["1 pound", "1 kilogram", "1 stone", "1 gram"], "A"),
    ("What is the only even prime number?", ["1", "2", "4", "0"], "B"),
]


def _fmt_choices(choices):
    return "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))


examples = [
    dspy.Example(
        question=q,
        choices=_fmt_choices(ch),
        answer=ans,
    ).with_inputs("question", "choices")
    for q, ch, ans in RAW
]

# train / val / test split
trainset = examples[:20]
valset = examples[20:30]
testset = examples[30:]

# --------------------------------------------------------------------------- #
# 3. Signature + module                                                       #
# --------------------------------------------------------------------------- #
class BasicQA(dspy.Signature):
    """Answer the multiple-choice question. Reply with the single letter (A, B, C, or D) of the correct option."""

    question = dspy.InputField()
    choices = dspy.InputField(desc="the answer options, one per line, each tagged with a letter")
    answer = dspy.OutputField(desc="the single letter of the correct choice")


class MultiChoices(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pred = dspy.ChainOfThought(BasicQA)

    def forward(self, question, choices):
        return self.pred(question=question, choices=choices)


# --------------------------------------------------------------------------- #
# 4. Metrics                                                                  #
# --------------------------------------------------------------------------- #
def _letter(text):
    """Pull the first A-D letter out of a model answer."""
    if text is None:
        return ""
    for ch in text.strip().upper():
        if ch in "ABCD":
            return ch
    return ""


def metric_simple(example, pred, trace=None):
    """bool metric for Evaluate + MIPROv2."""
    return _letter(pred.answer) == example.answer.strip().upper()


def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric: returns {'score', 'feedback'} so the reflection LM
    has something concrete to learn from."""
    got = _letter(pred.answer)
    want = gold.answer.strip().upper()
    correct = got == want
    if correct:
        fb = f"Correct. The answer is {want}."
    else:
        fb = (
            f"Incorrect. You answered '{pred.answer}' (parsed as '{got or 'none'}'), "
            f"but the correct option is '{want}'. "
            f"Question: {gold.question}\nChoices:\n{gold.choices}\n"
            f"Read each option carefully and output only the single correct letter."
        )
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=fb)


# --------------------------------------------------------------------------- #
# 5. Helpers                                                                  #
# --------------------------------------------------------------------------- #
evaluate = dspy.Evaluate(
    devset=testset,
    metric=metric_simple,
    num_threads=8,
    display_progress=True,
    display_table=0,
)


def score(program, label):
    t0 = time.time()
    res = evaluate(program)
    acc = res.score if hasattr(res, "score") else float(res)
    print(f"[{label}] test accuracy = {acc:.1f}%  ({time.time() - t0:.0f}s)")
    return acc


# --------------------------------------------------------------------------- #
# 6. Run: baseline -> MIPROv2 -> GEPA                                         #
# --------------------------------------------------------------------------- #
def main():
    results = {}

    print("\n=== Baseline (un-optimized ChainOfThought) ===")
    baseline = MultiChoices()
    results["baseline"] = score(baseline, "baseline")

    print("\n=== MIPROv2 ===")
    from dspy.teleprompt import MIPROv2

    mipro = MIPROv2(metric=metric_simple, auto="light", num_threads=8)
    mipro_prog = mipro.compile(
        MultiChoices(),
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False,
    )
    results["mipro"] = score(mipro_prog, "MIPROv2")
    mipro_prog.save("optimized_mipro.json")

    print("\n=== GEPA ===")
    from dspy.teleprompt import GEPA

    gepa = GEPA(
        metric=metric_feedback,
        auto="light",
        reflection_lm=reflection_lm,
        num_threads=8,
        track_stats=True,
    )
    gepa_prog = gepa.compile(
        MultiChoices(),
        trainset=trainset,
        valset=valset,
    )
    results["gepa"] = score(gepa_prog, "GEPA")
    gepa_prog.save("optimized_gepa.json")

    # --------------------------------------------------------------------- #
    print("\n" + "=" * 48)
    print("  Optimizer comparison  (test accuracy)")
    print("=" * 48)
    for name in ("baseline", "mipro", "gepa"):
        print(f"  {name:<10} {results[name]:5.1f}%")
    print("=" * 48)
    print("Saved prompts: optimized_mipro.json, optimized_gepa.json")


if __name__ == "__main__":
    main()
