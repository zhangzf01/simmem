#!/usr/bin/env python3
"""Deal rate and reward of held-out evaluation trajectories, as reported in the paper.

    python3 scripts/score_eval.py runs/eval_*/trajectories/eval/*/rollout_000000.jsonl

Paper scoring ("strict"): only an explicit donation marker in a persuadee turn counts ([DONATE $x], [GIVE $x] or
[$x]); the last marker of the dialogue is the donation. Reward = min(x / 2, 1); Deal = x > 0. A failed dialogue
(e.g. an API error) scores 0. The ``reward`` stored in the trajectory file additionally accepts natural-language
amounts ("I'll give 1") and is printed for reference.
"""
import argparse
import json
import math
import re

MARKER = re.compile(r"\[\s*(?:(?:DONATE|GIVE)\s*:?\s*\$?\s*|\$\s*)(\d+\.?\d*)\s*\]", re.I)


def strict_amount(messages):
    amounts = [float(m.group(1)) for msg in messages if msg.get("role") == "user"
               for m in MARKER.finditer(str(msg.get("content", "")))]
    return amounts[-1] if amounts else 0.0


def score(path):
    n = failed = 0
    stored = stored_deal = strict = strict_deal = 0.
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            failed += "FAILED" in str(row.get("status", ""))
            r = row.get("reward")
            r = float(r) if isinstance(r, (int, float)) and math.isfinite(r) else 0.
            stored += r
            stored_deal += r > 0
            amount = strict_amount(row.get("messages", []))
            strict += min(amount / 2., 1.)
            strict_deal += amount > 0
    return {"n": n, "failed": failed, "deal": strict_deal / n, "reward": strict / n,
            "stored_deal": stored_deal / n, "stored_reward": stored / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    print(f"{'file':<80} {'n':>4} {'fail':>4} {'Deal':>7} {'Reward':>7} | {'stored Deal':>11} {'stored R':>8}")
    for path in a.files:
        s = score(path)
        print(f"{path[-80:]:<80} {s['n']:>4} {s['failed']:>4} {100 * s['deal']:>6.1f}% {s['reward']:>7.3f} | "
              f"{100 * s['stored_deal']:>10.1f}% {s['stored_reward']:>8.3f}")


if __name__ == "__main__":
    main()
