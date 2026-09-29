"""Replay the memory loop over an existing run's trajectory files, step by step.

    python -m usim.simmem.offline --run runs/<run> --start 84 --end 92 --out /tmp/simmem \
        [--judge mock | --judge llm --judge-model <model id> --judge-base-url http://<host>:<port>/v1]

``--run`` is a training output directory (it holds ``trajectories/rollout_<step>.jsonl``). Each step renders the
current memory, then audits that step's saved groups exactly as the training hook does.
"""
import argparse
import asyncio
import json
import logging
import os

from .config import SimMemSettings
from .judge import MockJudge
from .updater import SimMemUpdater, make_llm_judge


def read_rows(path):
    rows = []
    with open(path) as f:
        for text in f:
            if not text.strip():
                continue
            r = json.loads(text)
            meta = r.get("metadata", {})
            group = meta.get("group") or str(meta.get("task_id"))
            rows.append({"task": str(meta.get("task_id", "")), "group": group, "reward": float(r.get("reward", 0.)),
                         "messages": r["messages"], "persona_id": meta.get("persuadee_speaker_id")})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="inclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=None, help="default: <repo>/data/p4g/corpus")
    ap.add_argument("--judge", default="mock", help="'mock' or 'llm'")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-api-key-var", default="OPENAI_API_KEY")
    ap.add_argument("--mean-threshold", type=float, default=.9)
    ap.add_argument("--std-threshold", type=float, default=.1)
    ap.add_argument("--max-audits-per-step", type=int, default=4)
    ap.add_argument("--init", default="")
    ap.add_argument("--max-notes", type=int, default=12)
    ap.add_argument("--frozen", action="store_true", help="fixed-memory control: render --init, never audit")
    ap.add_argument("--no-collapse-check", action="store_true", help="ablation: every gated group is audited as collapsed")
    ap.add_argument("--raw-context", action="store_true", help="ablation: store the flagged conversation, not a rule")
    ap.add_argument("--raw-max-notes", type=int, default=8)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    corpus = a.corpus or os.path.join(repo, "data", "p4g", "corpus")
    settings = SimMemSettings(enable=True, dir=a.out, mean_threshold=a.mean_threshold, std_threshold=a.std_threshold,
                              max_audits_per_step=a.max_audits_per_step, judge_model=a.judge_model or "mock",
                              judge_base_url=a.judge_base_url, judge_api_key_var=a.judge_api_key_var, init=a.init,
                              max_notes=a.max_notes, frozen=a.frozen, no_collapse_check=a.no_collapse_check,
                              raw_context=a.raw_context, raw_max_notes=a.raw_max_notes)
    if a.frozen:
        factory = None
    elif a.judge == "mock":
        factory = MockJudge
    else:
        if not a.judge_model:
            ap.error("--judge llm needs --judge-model")
        factory = lambda: make_llm_judge(settings)  # noqa: E731
    updater = SimMemUpdater(settings, factory, corpus)
    traj = os.path.join(a.run, "trajectories")
    steps = sorted(int(f[8:14]) for f in os.listdir(traj) if f.startswith("rollout_") and f.endswith(".jsonl"))
    for step in [s for s in steps if s >= a.start and (a.end is None or s <= a.end)]:
        block = updater.snapshot(step)
        metrics = asyncio.run(updater.after_step(step, read_rows(os.path.join(traj, f"rollout_{step:06d}.jsonl"))))
        print(json.dumps({"step": step, "snapshot_chars": len(block), **metrics}))
    print(f"{len(updater.memory.notes)} notes; current prompt block:\n"
          + (updater.memory.render(settings.render_notes) or "(empty)"))


if __name__ == "__main__":
    main()
