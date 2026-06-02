"""
MIPROv2 vs GEPA prompt-optimizer comparison on ALFWorld.

Mirrors the DSPy "Finetuning Agents" tutorial (dspy.ai/tutorials/games) ALFWorld
ReAct agent, but instead of weight finetuning it compares two *prompt*
optimizers — MIPROv2 and GEPA — on a weak student model.

Setup:
  * SAME model for all three roles (task / MIPRO prompt model / GEPA reflection):
    nvidia/llama-3.3-nemotron-super-49b-v1.5 (.env NeMoTronModel)
  * served OpenAI-compatibly via https://integrate.api.nvidia.com/v1

ALFWorld is a text household game: the agent gets a task ("put a clean mug in
the coffee machine"), sees admissible actions each step, and must reach the
goal within max_iters steps. Reward/success is 0/1 — lots of headroom for a
small model, which is exactly what makes the optimizer comparison meaningful.

IMPORTANT: the ALFWorld env pool uses multiprocess 'spawn', which re-imports
this module in every worker process. So the env pool (AlfWorld()) and anything
that touches it MUST be created inside main(), never at module top level — see
the `alfworld` global, which stays None on import and is only set in main().

    .venv/bin/python mipro_vs_gepa_alfworld.py            # full comparison
    .venv/bin/python mipro_vs_gepa_alfworld.py --smoke 3  # baseline on 3 tasks
"""

import os
import sys
import time

import dspy
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/dspy/.env")

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Same model for all three roles (task / MIPRO prompt model / GEPA reflection).
MODEL = "meta/llama-3.1-8b-instruct"

# knobs: identical test set scored across baseline/MIPRO/GEPA; small train/val
# feed the optimizers. Sized to finish < 1 hour at 40 RPM (see GEPA_METRIC_CALLS).
MAX_ITERS = 10
N_TRAIN = 8
N_VAL = 8
N_TEST = 15
GEPA_METRIC_CALLS = 50  # one metric call = one full rollout (up to MAX_ITERS LM calls)
NUM_THREADS = 1  # build.nvidia.com free tier is rate-limited (40 RPM); keep serial

task_lm = dspy.LM(
    f"openai/{MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=0.6,
    max_tokens=1000,
    num_retries=10,   # litellm exponential backoff on 429
)
dspy.configure(lm=task_lm)

# GEPA's reflection LM is the SAME model (just hotter, longer).
reflection_lm = dspy.LM(
    f"openai/{MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=1.0,
    max_tokens=4000,
    num_retries=10,
)

# Set ONLY inside main() (in the main process). Stays None when this module is
# re-imported by a spawned env worker, so the worker never builds another pool.
alfworld = None


# --------------------------------------------------------------------------- #
# Agent: one Predict that reasons + picks the next action, looped over the env #
# --------------------------------------------------------------------------- #
class Agent(dspy.Module):
    def __init__(self, max_iters=MAX_ITERS, verbose=False):
        super().__init__()
        self.max_iters = max_iters
        self.verbose = verbose
        self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")

    def forward(self, idx):
        reward = 0
        with alfworld.POOL.session() as env:
            trajectory = []
            task, info = env.init(idx)
            if self.verbose:
                print(f"Task: {task}", flush=True)

            for _ in range(self.max_iters):
                trajectory_ = "\n".join(trajectory)
                possible_actions = info["admissible_commands"][0] + ["think: ${...thoughts...}"]
                prediction = self.react(
                    task=task, trajectory=trajectory_, possible_actions=possible_actions
                )
                trajectory.append(f"> {prediction.action}")

                if prediction.action.startswith("think:"):
                    trajectory.append("OK.")
                    continue

                obs, reward, done, info = env.step(prediction.action)
                obs, reward, done = obs[0], reward[0], done[0]
                trajectory.append(obs)

                if done:
                    break

        return dspy.Prediction(trajectory="\n".join(trajectory), success=reward)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def metric_simple(example, pred, trace=None):
    """0/1 success — for Evaluate + MIPROv2."""
    return pred.success


def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric: the reflection LM reads `feedback` to rewrite the
    agent's instruction, so we describe what happened in the trajectory."""
    if pred.success:
        fb = "SUCCESS: the agent reached the goal."
    else:
        fb = (
            "FAILURE: the agent did not reach the goal within the step budget.\n"
            "Trajectory (actions taken and observations):\n"
            f"{pred.trajectory[-2000:]}\n"
            "Diagnose: did it locate the target object, navigate to it, and perform "
            "the required action in the right order? Propose a clearer step-by-step "
            "strategy (search likely receptacles, pick up/clean/heat as the task "
            "requires, then place at the destination). Avoid repeating useless actions."
        )
    return dspy.Prediction(score=float(pred.success), feedback=fb)


def score(program, label, evaluate):
    t0 = time.time()
    res = evaluate(program)
    acc = res.score if hasattr(res, "score") else float(res)
    print(f"[{label}] test success = {acc:.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    return acc


# --------------------------------------------------------------------------- #
def main():
    global alfworld
    from dspy.datasets.alfworld import AlfWorld

    alfworld = AlfWorld(max_threads=NUM_THREADS)

    # --smoke N : just run the baseline on N test tasks and exit
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    testset = alfworld.devset[:N_TEST]
    if smoke:
        sub = alfworld.devset[:smoke]
        ev = dspy.Evaluate(devset=sub, metric=metric_simple, num_threads=NUM_THREADS,
                           display_progress=True, provide_traceback=True)
        score(Agent(verbose=True), f"baseline-smoke-{smoke}", ev)
        return

    trainset = alfworld.trainset[:N_TRAIN]
    valset = alfworld.trainset[N_TRAIN : N_TRAIN + N_VAL]
    evaluate = dspy.Evaluate(devset=testset, metric=metric_simple, num_threads=NUM_THREADS,
                             display_progress=True, display_table=0, provide_traceback=True)

    results = {}

    print("\n=== Baseline (un-optimized agent) ===", flush=True)
    results["baseline"] = score(Agent(), "baseline", evaluate)

    print("\n=== MIPROv2 ===", flush=True)
    from dspy.teleprompt import MIPROv2

    mipro = MIPROv2(metric=metric_simple, auto="light", num_threads=NUM_THREADS,
                    prompt_model=reflection_lm)
    mipro_prog = mipro.compile(
        Agent(), trainset=trainset, valset=valset,
        max_bootstrapped_demos=1, max_labeled_demos=0,
        minibatch_size=N_VAL,
        requires_permission_to_run=False,
    )
    results["mipro"] = score(mipro_prog, "MIPROv2", evaluate)
    mipro_prog.save("optimized_mipro_alfworld.json")

    print("\n=== GEPA ===", flush=True)
    from dspy.teleprompt import GEPA

    gepa = GEPA(metric=metric_feedback, max_metric_calls=GEPA_METRIC_CALLS,
                reflection_lm=reflection_lm, num_threads=NUM_THREADS, track_stats=True)
    gepa_prog = gepa.compile(Agent(), trainset=trainset, valset=valset)
    results["gepa"] = score(gepa_prog, "GEPA", evaluate)
    gepa_prog.save("optimized_gepa_alfworld.json")

    print("\n" + "=" * 48)
    print("  ALFWorld optimizer comparison (test success)")
    print("=" * 48)
    for name in ("baseline", "mipro", "gepa"):
        print(f"  {name:<10} {results[name]:5.1f}%")
    print("=" * 48, flush=True)


if __name__ == "__main__":
    main()
