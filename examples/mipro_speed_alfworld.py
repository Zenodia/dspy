"""
"How fast does MIPRO solve it vs baseline" on a SINGLE hard ALFWorld game.

Approach (kept deliberately simple, per textgame_exp.ipynb):
  * A ReAct agent: one dspy.Predict that, each step, sees the task, a WINDOW of
    the last few (action, observation) turns, and the admissible actions, then
    picks the next action. No separate "agent memory" add-on is needed — the
    sliding window carries the recent game state.
  * The window bounds the prompt so it never overflows the context. If the agent
    still hasn't finished within MAX_ITERS steps (or any context/API error is
    hit mid-episode), the game is counted as a LOSS (success = 0).

We fix ONE hard game (a "find two X and put them in Y" task) and measure, for
the SAME model (nemotron-49b):
  * baseline agent          -> solved? in how many steps?
  * MIPROv2-optimized agent -> solved? in how many steps?
"Faster" = fewer env steps to reach the goal.

A small phase-tagged on-disk log is written per rollout (train vs eval) so the
two phases are separable on disk — but it is a record only, not fed to the LLM.

Same model for everything: nvidia/llama-3.3-nemotron-super-49b-v1.5 (.env).
Served OpenAI-compatibly via https://integrate.api.nvidia.com/v1.

Run as a FILE (the ALFWorld env pool uses multiprocess 'spawn'):
    .venv/bin/python mipro_speed_alfworld.py
    .venv/bin/python mipro_speed_alfworld.py --idx 1650 --baseline-only
"""

import os
import sys
import time

import dspy
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/dspy/.env")

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
BASE_URL = "https://integrate.api.nvidia.com/v1"
# Non-reasoning instruct model (the textgame_exp.ipynb choice). Despite more
# params than nemotron-49b, it is MUCH faster here: no long hidden think-trace
# per step (~3-5s/step vs ~31s/step), which is what makes a 3-optimizer smoke
# on a single game finish in ~1 hour.
MODEL = "meta/llama-3.3-70b-instruct"

MAX_ITERS = 12          # step budget; not solved within this => loss
WINDOW = 8              # keep last N (action, obs) turns in the prompt
TARGET_IDX = 1650       # single target game
GEPA_METRIC_CALLS = 5   # tiny GEPA budget for the smoke (one call = one rollout)
MEM_DIR = "/home/ubuntu/dspy/alfworld_memory"

INSTRUCTIONS = (
    "Interact with a simulated household to achieve a high-level goal. Plan, "
    "track subgoals, and reason about likely locations for common items (e.g. a "
    "saltshaker is likely in a cabinet, on a countertop, or a diningtable). "
    "Explore systematically (check candidate receptacles one by one, opening "
    "closed ones). For 'find two X' tasks, remember you must place BOTH. Reply "
    "with exactly one admissible action."
)

task_lm = dspy.LM(
    f"openai/{MODEL}", api_key=NVIDIA_API_KEY, api_base=BASE_URL,
    temperature=0.6, max_tokens=4000, num_retries=10,
)
dspy.configure(lm=task_lm)

reflection_lm = dspy.LM(
    f"openai/{MODEL}", api_key=NVIDIA_API_KEY, api_base=BASE_URL,
    temperature=1.0, max_tokens=8000, num_retries=10,
)

alfworld = None         # built only in main() — see spawn note in the docstring
RUN_PHASE = "baseline"  # baseline | train | eval — tags the on-disk memory
EPISODE = 0


# --------------------------------------------------------------------------- #
# Lightweight phase-tagged on-disk log (record only; NOT fed to the LLM)       #
# --------------------------------------------------------------------------- #
class DiskLog:
    def __init__(self, path, task, phase, episode):
        self.path, self.turn = path, 0
        with open(path, "w") as f:
            f.write("@@@MEMORY_LOG_START@@@\n")
            f.write(f"@PHASE:{phase}@\n@EPISODE:{episode:04d}@\n@TASK:{task}@\n>>>TURNS_START<<<\n")

    def log_turn(self, action, obs, reward, done):
        self.turn += 1
        with open(self.path, "a") as f:
            f.write(f"<<<TURN:{self.turn:04d}>>>\n")
            f.write(f">>>ACT:{self.turn:04d}>>>\n{action}\n<<<ACT:{self.turn:04d}<<<\n")
            f.write(f">>>OBS:{self.turn:04d}>>>\n{obs}\n<<<OBS:{self.turn:04d}<<<\n")
            f.write(f"@REWARD:{reward}@ @DONE:{done}@\n<<<END_TURN:{self.turn:04d}>>>\n\n")


def clean_action(raw, possible):
    """Reasoning models leak chain-of-thought and DSPy field delimiters into the
    action field. Strip that and snap the action to a valid admissible command."""
    a = (raw or "").split("[[")[0].strip().splitlines()
    a = (a[-1] if a else "").strip(" `\"'.>").strip()
    low = a.lower()
    if low.startswith("think:"):
        return a
    for cmd in possible:                       # exact
        if cmd.lower() == low:
            return cmd
    for cmd in possible:                       # substring either way
        if cmd != "think: ${...thoughts...}" and (low in cmd.lower() or cmd.lower() in low):
            return cmd
    return a


class Agent(dspy.Module):
    def __init__(self, max_iters=MAX_ITERS, window=WINDOW, verbose=False):
        super().__init__()
        self.max_iters = max_iters
        self.window = window
        self.verbose = verbose
        self.react = dspy.Predict(
            dspy.Signature("task, trajectory, possible_actions: list[str] -> action",
                           INSTRUCTIONS)
        )

    def forward(self, idx):
        global EPISODE
        EPISODE += 1
        reward, steps = 0, 0
        try:
            with alfworld.POOL.session() as env:
                task, info = env.init(idx)
                log = DiskLog(os.path.join(MEM_DIR, f"{RUN_PHASE}__game{idx}__ep{EPISODE:04d}.txt"),
                              task, RUN_PHASE, EPISODE)
                trajectory = []
                if self.verbose:
                    print(f"Task: {task.splitlines()[-1].strip()}", flush=True)

                for _ in range(self.max_iters):
                    possible = info["admissible_commands"][0] + ["think: ${...thoughts...}"]
                    window = "\n".join(trajectory[-2 * self.window:])  # last N turns
                    pred = self.react(task=task, trajectory=window, possible_actions=possible)
                    action = clean_action(pred.action, possible)
                    trajectory.append(f"> {action}")

                    if action.startswith("think:"):
                        trajectory.append("OK.")
                        continue

                    obs, reward, done, info = env.step(action)
                    obs, reward, done = obs[0], reward[0], done[0]
                    trajectory.append(obs)
                    log.log_turn(action, obs, reward, done)
                    steps += 1
                    if self.verbose:
                        print(f"  [{steps}] {action} -> {obs}", flush=True)
                    if done:
                        break
        except Exception as e:
            # context overflow / API error mid-episode => count as a loss
            print(f"  [rollout error -> loss] {type(e).__name__}: {str(e)[:140]}", flush=True)
            return dspy.Prediction(success=0, steps=self.max_iters, solved=False, trajectory="")

        return dspy.Prediction(success=reward, steps=steps if reward else self.max_iters,
                               solved=bool(reward), trajectory="\n".join(trajectory))


def metric_speed(example, pred, trace=None):
    """Reward FAST success: 1.0 for solving in 1 step, ->0 as steps grow, 0 if
    unsolved. MIPRO maximizes this, so it is pushed to solve in fewer steps."""
    if not pred.solved:
        return 0.0
    return max(0.0, 1.0 - (pred.steps - 1) / MAX_ITERS)


def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric: the reflection LM reads `feedback` to rewrite the
    agent's instruction."""
    if pred.solved:
        fb = f"SUCCESS in {pred.steps} steps. Could it be done in fewer?"
    else:
        fb = ("FAILURE: did not finish within the step budget — likely looped or "
              "explored unsystematically.\nTrajectory tail:\n"
              f"{pred.trajectory[-1500:]}\n"
              "Propose a clearer strategy: track which receptacles were already "
              "checked, go straight to likely locations, then place the item.")
    return dspy.Prediction(score=metric_speed(gold, pred), feedback=fb)


def run_once(program, idx, label, phase):
    global RUN_PHASE
    RUN_PHASE = phase
    t0 = time.time()
    pred = program(idx=idx)
    dt = time.time() - t0
    print(f"[{label}] solved={pred.solved}  steps={pred.steps}  ({dt:.0f}s)", flush=True)
    return {"solved": pred.solved, "steps": pred.steps, "seconds": round(dt)}


def main():
    global alfworld, RUN_PHASE
    os.makedirs(MEM_DIR, exist_ok=True)
    from dspy.datasets.alfworld import AlfWorld

    alfworld = AlfWorld(max_threads=2)

    idx = TARGET_IDX
    if "--idx" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--idx") + 1])
    print(f"Target game idx = {idx}  (MAX_ITERS={MAX_ITERS}, WINDOW={WINDOW})", flush=True)

    print("\n=== Baseline (single game) ===", flush=True)
    base = run_once(Agent(verbose=True), idx, "baseline", phase="baseline")

    if "--baseline-only" in sys.argv:
        return

    print("\n=== MIPROv2 compile (train-phase rollouts) ===", flush=True)
    RUN_PHASE = "train"
    from dspy.teleprompt import MIPROv2

    trainset = [e for e in alfworld.trainset[:4] if e["idx"] != idx][:2]
    valset = alfworld.trainset[4:5]
    mipro = MIPROv2(metric=metric_speed, auto=None, num_candidates=2, num_threads=1,
                    prompt_model=reflection_lm, max_bootstrapped_demos=1)
    mipro_prog = mipro.compile(
        Agent(), trainset=trainset, valset=valset,
        num_trials=2, minibatch=False, max_labeled_demos=0,
        requires_permission_to_run=False,
    )
    mipro_prog.save("optimized_mipro_speed.json")

    print("\n=== MIPROv2 (single game, eval-phase) ===", flush=True)
    mipro_prog.verbose = True
    mp = run_once(mipro_prog, idx, "MIPROv2", phase="eval")

    print("\n=== GEPA compile (train-phase rollouts) ===", flush=True)
    RUN_PHASE = "train"
    from dspy.teleprompt import GEPA

    gepa = GEPA(metric=metric_feedback, max_metric_calls=GEPA_METRIC_CALLS,
                reflection_lm=reflection_lm, num_threads=1, track_stats=True)
    gepa_prog = gepa.compile(Agent(), trainset=trainset, valset=valset)
    gepa_prog.save("optimized_gepa_speed.json")

    print("\n=== GEPA (single game, eval-phase) ===", flush=True)
    gepa_prog.verbose = True
    gp = run_once(gepa_prog, idx, "GEPA", phase="eval")

    print("\n" + "=" * 56)
    print(f"  ALFWorld single-game SPEED: baseline vs MIPROv2 vs GEPA (idx={idx})")
    print("=" * 56)
    print(f"  {'':<10}{'solved':>9}{'steps':>8}{'seconds':>9}")
    for name, r in (("baseline", base), ("MIPROv2", mp), ("GEPA", gp)):
        print(f"  {name:<10}{str(r['solved']):>9}{r['steps']:>8}{r['seconds']:>9}")
    print("=" * 56, flush=True)


if __name__ == "__main__":
    main()
