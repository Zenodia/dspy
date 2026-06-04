"""
GEPA-optimized Haystack memory agent on ALFWorld.

A *hybrid* agent for the ALFWorld text game:

  * Haystack is the specialized agentic layer (NVIDIA-served):
      - NvidiaDocumentEmbedder + InMemoryDocumentStore  -> write memory items
      - NvidiaTextEmbedder + InMemoryEmbeddingRetriever  -> recall top-k memories
      - NvidiaChatGenerator (moonshotai/kimi-k2.6) + a per-step Tool whose enum
        is the current admissible_commands -> FORCED-valid next action via tool
        calling. kimi "thinking" mode plans the long horizon before committing.

  * DSPy owns the one optimizable surface: a dspy.Predict summarizer that turns
    (task, recent_obs, history) into a memory_item. GEPA evolves ONLY that
    predictor's instruction. Experimental question: can GEPA learn to summarize
    observations into memory items that let a fixed reasoner solve in fewer steps?

Objective (why fewest-steps falls out for free)
------------------------------------------------
ALFWorld is an episodic MDP with a sparse terminal reward r_T = 1 on success.
The Q-learning / discounted-return target

    Q(s,a) = r + gamma * max_a' Q(s',a')          (Bellman, gamma in (0,1))
    G_0     = sum_t gamma^t r_t  =  gamma^T * 1[success]

collapses (sparse terminal reward) to gamma^T. Because gamma < 1, gamma^T is
strictly decreasing in T, so maximizing E[G_0] == reaching the goal in the
FEWEST steps. We pick gamma so the discount horizon equals the step budget H,
i.e. return decays to exactly 1/e at the limit:

    gamma = e^(-1/(H-1))   ->  gamma^0 = 1.0 (1-step solve),  gamma^(H-1) = 1/e

That gives a smooth, theory-grounded gradient across the whole 50-step budget
with no hand-tuned penalty constant. GEPA maximizes this scalar directly.

Models (all served OpenAI-compatibly via integrate.api.nvidia.com -> ONE 40 RPM
pool; kimi adds tokens+latency, not call count):
  * reasoner / GEPA reflection : moonshotai/kimi-k2.6   (.env VLM_Model)
  * summarizer (GEPA target)   : nvidia/llama-3.3-nemotron-super-49b-v1
  * embedder                   : nvidia/llama-3.2-nv-embedqa-1b-v2

IMPORTANT: ALFWorld's env pool uses multiprocess 'spawn', which re-imports this
module in every worker. The env pool (AlfWorld()) and the Haystack singletons
must therefore only be built inside main()/forward() in the MAIN process -- see
the `alfworld` global (stays None on import) and the lazy `_HAYSTACK` cache.

    .venv/bin/python gepa_haystack_alfworld.py            # full baseline vs GEPA
    .venv/bin/python gepa_haystack_alfworld.py --smoke 2  # pipeline check, 2 tasks
"""

import json
import math
import os
import sys
import time

import dspy
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

colorama_init(autoreset=True)  # color-coded trace; resets style after each print

from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.tools import Tool
from haystack.utils import Secret
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
from haystack_integrations.components.embedders.nvidia import (
    NvidiaDocumentEmbedder,
    NvidiaTextEmbedder,
)

load_dotenv("/home/ubuntu/dspy/.env")

NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
BASE_URL = "https://integrate.api.nvidia.com/v1"

# --- models ---------------------------------------------------------------- #
REASONER_MODEL = "moonshotai/kimi-k2.6"                       # .env VLM_Model
SUMMARIZER_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"   # GEPA target
# (.env NeMoTronModel is the v1.5 variant; swap here if v1 is unavailable.)
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"  # embedqa, live + callable on this account (1024-d)

# --- knobs (sized for the 40 RPM free tier; the full run is multi-hour) ---- #
H = MAX_ITERS = 50                       # user mandate: 50 steps max per game
GAMMA = math.exp(-1.0 / (H - 1))         # ~0.9798: discount horizon == budget
N_TRAIN = 6
N_VAL = 6
N_TEST = 10
GEPA_METRIC_CALLS = 30                   # one metric call == one full rollout
NUM_THREADS = 1                          # 40 RPM -> stay serial
TOP_K_MEMORY = 3
MIN_INTERVAL = 2.2                       # s between Haystack NVIDIA calls (~27/min headroom)

# --- color-coded trace logging -------------------------------------------- #
# VERBOSE is a module global (not an instance attr) so evaluate_program can turn
# the trace ON for baseline/final eval but keep GEPA's internal rollouts QUIET.
VERBOSE = False


def _log(tag, text, color, dim=False):
    if not VERBOSE:
        return
    style = Style.DIM if dim else Style.BRIGHT
    print(f"{color}{style}{tag:<9}{Style.RESET_ALL}{color}{'' if dim else Style.NORMAL}"
          f"{text}{Style.RESET_ALL}", flush=True)


def parse_goal(task):
    """Pull the clean objective out of ALFWorld's welcome blurb."""
    if "Your task is to:" in task:
        return task.split("Your task is to:")[-1].strip()
    return task.strip()


def log_goal(goal):
    # Always printed (not VERBOSE-gated): the objective of THIS game.
    print(f"{Fore.CYAN}{Style.BRIGHT}GOAL     {goal}{Style.RESET_ALL}", flush=True)


def log_task(task):
    _log("INTRO", task, Fore.WHITE, dim=True)


def log_step(n):
    if VERBOSE:
        print(f"{Fore.WHITE}{Style.DIM}{'-' * 70}  step {n}{Style.RESET_ALL}", flush=True)


def log_obs(obs):
    _log("OBS", obs, Fore.WHITE, dim=True)


def log_memory(item):
    _log("MEMORY", item, Fore.YELLOW)          # summarizer's reasoning about state


def log_recall(items):
    body = "  |  ".join(items) if items else "(none yet)"
    _log("RECALL", body, Fore.MAGENTA)         # top-k memories pulled by retriever


def log_action(action):
    _log("ACTION", action, Fore.GREEN)         # move kimi committed to via tool call


def log_result(success, steps, ret):
    color = Fore.GREEN if success else Fore.RED
    word = "SOLVED" if success else "FAILED"
    print(f"{color}{Style.BRIGHT}{word}{Style.RESET_ALL}{color} in {steps} steps  "
          f"(return={ret:.3f}){Style.RESET_ALL}", flush=True)


# --- DSPy LMs (summarizer is the optimized role; reflection is kimi) -------- #
summarizer_lm = dspy.LM(
    f"openai/{SUMMARIZER_MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=0.6,
    max_tokens=1000,
    num_retries=10,                      # litellm exponential backoff on 429
)
dspy.configure(lm=summarizer_lm)

reflection_lm = dspy.LM(
    f"openai/{REASONER_MODEL}",
    api_key=NVIDIA_API_KEY,
    api_base=BASE_URL,
    temperature=1.0,
    max_tokens=8000,
    num_retries=10,
)

# Set ONLY inside main() (main process). Stays None on worker re-import.
alfworld = None

# Lazy Haystack singletons (built on first forward, in the main process, so they
# survive 'spawn' re-import and avoid being deep-copied by GEPA via self.*).
_HAYSTACK = {}


def _throttle():
    """Crude global rate limiter: >= MIN_INTERVAL between Haystack NVIDIA calls.
    Combined with NUM_THREADS=1 and num_retries on the DSPy side, keeps the
    shared 40 RPM endpoint from 429-ing. Not exact (DSPy calls aren't counted),
    hence the conservative interval + retries."""
    last = _HAYSTACK.get("_last", 0.0)
    dt = time.monotonic() - last
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _HAYSTACK["_last"] = time.monotonic()


def _reasoner():
    if "gen" not in _HAYSTACK:
        _HAYSTACK["gen"] = NvidiaChatGenerator(
            model=REASONER_MODEL,
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            api_base_url=BASE_URL,
            max_retries=8,   # backoff on 429 instead of dropping to a junk fallback
        )
    return _HAYSTACK["gen"]


def _doc_embedder():
    if "doc" not in _HAYSTACK:
        e = NvidiaDocumentEmbedder(
            model=EMBED_MODEL, api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            api_url=BASE_URL,
        )
        e.warm_up()
        _HAYSTACK["doc"] = e
    return _HAYSTACK["doc"]


def _text_embedder():
    if "txt" not in _HAYSTACK:
        e = NvidiaTextEmbedder(
            model=EMBED_MODEL, api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            api_url=BASE_URL,
        )
        e.warm_up()
        _HAYSTACK["txt"] = e
    return _HAYSTACK["txt"]


REASON_SYS = (
    "You are an expert agent playing ALFWorld, a text-based household game. "
    "Your goal is to COMPLETE THE TASK IN AS FEW STEPS AS POSSIBLE. You are "
    "given the task, a memory of salient facts gathered so far, and the list of "
    "admissible actions for this step. Call the `take_action` tool with exactly "
    "ONE action chosen from the admissible list. Prefer actions that make direct "
    "progress toward the goal; do not revisit already-explored receptacles or "
    "repeat actions that produced no new information."
)


def _make_action_tool(admissible):
    """One Tool, rebuilt each step: its `action` enum == the admissible commands
    right now, so a forced tool call is guaranteed to be a legal move."""
    return Tool(
        name="take_action",
        description="Choose the single best next action toward completing the task.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(admissible),
                    "description": "The next action, copied verbatim from the admissible list.",
                }
            },
            "required": ["action"],
        },
        function=lambda action: action,
    )


def _reason(task, memory_ctx, admissible, recent_actions=()):
    """kimi picks a guaranteed-legal next action via a forced tool call."""
    _throttle()
    tool = _make_action_tool(admissible)
    messages = [
        ChatMessage.from_system(REASON_SYS),
        ChatMessage.from_user(
            f"Task: {task}\n\nMemory (recalled, most relevant first):\n"
            f"{memory_ctx or '(none yet)'}\n\nAdmissible actions:\n"
            + "\n".join(f"  - {a}" for a in admissible)
        ),
    ]
    try:
        out = _reasoner().run(
            messages=messages,
            tools=[tool],
            generation_kwargs={
                "tool_choice": "required",
                # kimi "thinking" is a non-OpenAI field -> must ride in extra_body,
                # else the OpenAI client rejects it as an unknown kwarg.
                "extra_body": {"chat_template_kwargs": {"thinking": True}},
                "max_tokens": 6000,
                "temperature": 0.6,
            },
        )
        reply = out["replies"][0]
        if reply.tool_calls:
            a = reply.tool_calls[0].arguments.get("action")
            if a in admissible:
                return a
        # fallback: substring match against the free text, else first action
        text = reply.text or ""
        for a in admissible:
            if a in text:
                return a
    except Exception as e:  # noqa: BLE001 - degrade gracefully, keep the rollout alive
        print(f"  {Fore.RED}[reason error: {e}]{Style.RESET_ALL}", flush=True)
    # Fallback (only if the tool call genuinely failed): avoid repeating a recent
    # action so a transient failure can't trap the agent in an examine/look loop.
    fresh = [a for a in admissible if a not in recent_actions]
    pool = fresh or admissible
    explore = [a for a in pool if a.startswith(("go to", "open"))]
    return (explore or pool)[0]


# --------------------------------------------------------------------------- #
# Agent: DSPy summarizer (GEPA target) + Haystack memory & reasoner            #
# --------------------------------------------------------------------------- #
class HaystackMemoryAgent(dspy.Module):
    def __init__(self, max_iters=MAX_ITERS, verbose=False):
        super().__init__()
        self.max_iters = max_iters
        self.verbose = verbose
        # The ONE GEPA-optimizable predictor.
        self.summarize = dspy.Predict(
            "task, recent_obs, history -> memory_item"
        )

    def forward(self, idx):
        # Per-episode memory store so memories never leak across tasks.
        store = InMemoryDocumentStore()
        retriever = InMemoryEmbeddingRetriever(document_store=store)

        reward = 0
        steps = 0
        with alfworld.POOL.session() as env:
            trajectory = []
            task, info = env.init(idx)
            goal = parse_goal(task)
            log_goal(goal)   # objective of this game, recorded before the trace
            log_task(task)

            for _ in range(self.max_iters):
                admissible = info["admissible_commands"][0]
                recent_obs = trajectory[-1] if trajectory else "(game start)"
                log_step(steps + 1)
                log_obs(recent_obs)

                # 1. DSPy summarizer -> memory item (the GEPA-optimized step).
                #    This memory_item IS the agent's surfaced reasoning about the
                #    current state. (kimi's own thinking-mode deliberation in step 4
                #    runs internally and is NOT exposed by nvidia-haystack 0.3.0.)
                mem = self.summarize(
                    task=task,
                    recent_obs=recent_obs,
                    history="\n".join(trajectory[-6:]),
                ).memory_item
                log_memory(mem)

                # 2. embed + store the NEW memory item only (1 doc-embed call)
                _throttle()
                docs = _doc_embedder().run(documents=[Document(content=mem)])["documents"]
                store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

                # 3. recall top-k memories relevant to task+current obs
                _throttle()
                q_emb = _text_embedder().run(text=f"{task}\n{recent_obs}")["embedding"]
                top = retriever.run(query_embedding=q_emb, top_k=TOP_K_MEMORY)["documents"]
                recalled = [d.content for d in top]
                memory_ctx = "\n".join(f"- {c}" for c in recalled)
                log_recall(recalled)

                # 4. kimi reasons + emits a forced-valid action
                recent = [t[2:] for t in trajectory if t.startswith("> ")][-4:]
                action = _reason(task, memory_ctx, admissible, recent_actions=recent)
                trajectory.append(f"> {action}")
                log_action(action)

                # 5. step the env
                obs, reward, done, info = env.step(action)
                obs, reward, done = obs[0], reward[0], done[0]
                trajectory.append(obs)
                steps += 1

                if done:
                    break

        log_result(bool(reward), steps, discounted_return(reward, steps))
        return dspy.Prediction(
            goal=goal, trajectory="\n".join(trajectory), success=reward, steps=steps
        )


# --------------------------------------------------------------------------- #
# Metric: discounted terminal return (the objective derived above)             #
# --------------------------------------------------------------------------- #
def discounted_return(success, steps):
    return (GAMMA ** (steps - 1)) if success else 0.0


def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric. score == discounted return; feedback teaches the
    reflection LM to rewrite the summarizer so the fixed reasoner finishes in
    fewer steps."""
    ret = discounted_return(pred.success, pred.steps)
    if pred.success:
        fb = (
            f"SUCCESS in {pred.steps} steps (discounted return = {ret:.3f}; "
            f"1.0 == solved in 1 step). FEWER steps -> HIGHER return. The memory "
            f"items should have surfaced the target object's location and the "
            f"required sub-steps sooner. What earlier observation, if summarized "
            f"more sharply into memory, would have cut wasted steps?"
        )
    else:
        fb = (
            "FAILURE: goal not reached within the 50-step budget (return = 0).\n"
            "Trajectory (actions and observations):\n"
            f"{pred.trajectory[-2000:]}\n"
            "Diagnose from the trajectory: were memory items vague, redundant, or "
            "missing the target location / cleaning / heating sub-goal? Rewrite the "
            "summarizer instruction so each memory item captures (a) where things "
            "were found, (b) what sub-goal remains, (c) which actions already "
            "failed -- so the reasoner stops repeating dead ends."
        )
    return dspy.Prediction(score=ret, feedback=fb)


# --------------------------------------------------------------------------- #
def evaluate_program(program, testset, label):
    """Manual serial eval reporting all three numbers the comparison cares about:
    the discounted-return objective, raw success %, and avg steps-to-solve."""
    global VERBOSE
    prev_verbose = VERBOSE
    VERBOSE = True  # show the full color trace during eval; restored below
    t0 = time.time()
    succ, steps_solved, rets, records = [], [], [], []
    for i, ex in enumerate(testset):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}===== [{label}] task {i + 1}/{len(testset)} "
              f"====={Style.RESET_ALL}", flush=True)
        pred = program(**ex.inputs())
        ret = discounted_return(pred.success, pred.steps)
        rets.append(ret)
        succ.append(1.0 if pred.success else 0.0)
        if pred.success:
            steps_solved.append(pred.steps)
        # per-game record: the GOAL is registered alongside the outcome + trace
        records.append({
            "task": i + 1,
            "idx": ex.inputs().get("idx"),
            "goal": getattr(pred, "goal", None),
            "success": int(bool(pred.success)),
            "steps": pred.steps,
            "return": round(ret, 4),
            "trajectory": pred.trajectory,
        })
    VERBOSE = prev_verbose
    avg_ret = sum(rets) / len(rets)
    succ_pct = 100.0 * sum(succ) / len(succ)
    avg_steps = (sum(steps_solved) / len(steps_solved)) if steps_solved else float("nan")
    summary = {"return": avg_ret, "success": succ_pct, "avg_steps": avg_steps}

    # Dump goal-tagged per-game results to its own file (distinct from the
    # optimized-prompt .json, which holds no task info).
    out_path = f"results_{label}.json"
    with open(out_path, "w") as f:
        json.dump({"label": label, "gamma": GAMMA, "summary": summary,
                   "games": records}, f, indent=2)
    print(f"[{label}] return={avg_ret:.3f}  success={succ_pct:.1f}%  "
          f"avg_steps(solved)={avg_steps:.1f}  ({time.time() - t0:.0f}s)  "
          f"-> {out_path}", flush=True)
    return summary


def main():
    global alfworld
    from dspy.datasets.alfworld import AlfWorld

    alfworld = AlfWorld(max_threads=NUM_THREADS)

    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    if smoke:
        sub = alfworld.devset[:smoke]
        evaluate_program(HaystackMemoryAgent(verbose=True), sub, f"baseline-smoke-{smoke}")
        return

    testset = alfworld.devset[:N_TEST]
    trainset = alfworld.trainset[:N_TRAIN]
    valset = alfworld.trainset[N_TRAIN : N_TRAIN + N_VAL]

    results = {}

    print("\n=== Baseline (un-optimized summarizer) ===", flush=True)
    results["baseline"] = evaluate_program(HaystackMemoryAgent(), testset, "baseline")

    print("\n=== GEPA (optimizing the summarizer) ===", flush=True)
    from dspy.teleprompt import GEPA

    gepa = GEPA(
        metric=metric_feedback,
        max_metric_calls=GEPA_METRIC_CALLS,
        reflection_lm=reflection_lm,
        num_threads=NUM_THREADS,
        track_stats=True,
    )
    gepa_prog = gepa.compile(HaystackMemoryAgent(), trainset=trainset, valset=valset)
    gepa_prog.save("optimized_gepa_haystack_alfworld.json")
    results["gepa"] = evaluate_program(gepa_prog, testset, "GEPA")

    print("\n" + "=" * 60)
    print("  GEPA + Haystack memory agent on ALFWorld")
    print("  objective = discounted return  gamma = e^(-1/49) ~ {:.4f}".format(GAMMA))
    print("=" * 60)
    print(f"  {'':<10}{'return':>10}{'success%':>12}{'avg_steps':>12}")
    for name in ("baseline", "gepa"):
        r = results[name]
        print(f"  {name:<10}{r['return']:>10.3f}{r['success']:>11.1f}%{r['avg_steps']:>12.1f}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
