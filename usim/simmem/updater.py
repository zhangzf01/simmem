"""Per-step driver: snapshot before the step, judge high-reward groups after it, append rules."""
import asyncio
import json
import logging
import os
import re
import statistics
import time

from .config import SimMemSettings
from .judge import Judge
from .memory import Memory

logger = logging.getLogger(__name__)
_UPDATERS = {}


def load_personas(corpus_dir):
    users, ee = {}, {}
    try:
        with open(os.path.join(corpus_dir, "users.json")) as f:
            users = json.load(f)
        with open(os.path.join(corpus_dir, "conversations.json")) as f:
            ee = {k: v.get("user_ee") for k, v in json.load(f).items()}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[simmem] persona corpus unavailable (%s)", exc)
    return users, ee


def suspicious_groups(rows, settings):
    """Groups with mean reward >= threshold and population std <= threshold, highest mean first."""
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"group": r["group"], "task": str(r["task"]),
                                           "persona_id": r.get("persona_id"), "rewards": [], "dialogues": []})
        g["rewards"].append(float(r["reward"]))
        g["dialogues"].append([{"role": m["role"], "content": m.get("content") or ""}
                               for m in r["messages"] if m.get("role") in ("assistant", "user")])
    out = []
    for g in groups.values():
        g["mean"] = statistics.fmean(g["rewards"])
        g["std"] = statistics.pstdev(g["rewards"]) if len(g["rewards"]) > 1 else 0.
        if len(g["rewards"]) >= 2 and g["mean"] >= settings.mean_threshold and g["std"] <= settings.std_threshold:
            out.append(g)
    out.sort(key=lambda g: (-g["mean"], g["std"], g["group"]))
    return out, len(groups)


def pick_dialogue(group, script=""):
    """Raw-context ablation: the conversation the judge cites first (``C<i>`` in ``script``), else the highest-reward
    one, shortest on ties (the canonical run of the script). Returns (index, dialogue)."""
    n = len(group["dialogues"])
    for m in re.findall(r"\bC(\d+)\b", script or ""):
        if int(m) < n:
            return int(m), group["dialogues"][int(m)]
    i = min(range(n), key=lambda k: (-group["rewards"][k], sum(len(x["content"]) for x in group["dialogues"][k]), k))
    return i, group["dialogues"][i]


class SimMemUpdater:
    def __init__(self, settings, judge_factory, corpus_dir=""):
        """``judge_factory()`` returns an object with ``judge(persona, dialogues, notes)``; called once per
        step so the API client binds to that step's event loop."""
        self.settings = settings.validate()
        self.memory = Memory.load(settings.dir, settings.init)
        self.judge_factory = judge_factory
        self.users, self.task_ee = load_personas(corpus_dir) if corpus_dir else ({}, {})
        logger.info("[simmem] memory at %s: %d notes", settings.dir, len(self.memory.notes))

    def snapshot(self, step):
        block = self.memory.render(self.settings.render_notes)
        self._log("snapshot", step=step, notes=len(self.memory.notes), chars=len(block))
        return block

    def persona_for(self, g):
        """The persuadee persona record of the group (P4G users.json)."""
        pid = g.get("persona_id") or self.task_ee.get(g["task"])
        return self.users.get(pid) or {"persuadee_speaker_id": pid}

    async def after_step(self, step, rows):
        """``rows``: one dict per trajectory of the step with keys task, group, reward, messages, persona_id."""
        t0 = time.time()
        suspicious, n_groups = suspicious_groups(rows, self.settings)
        metrics = {"simmem/notes": float(len(self.memory.notes)), "simmem/suspicious_groups": float(len(suspicious)),
                   "simmem/suspicious_frac": (len(suspicious) / n_groups) if n_groups else 0.}
        chosen = suspicious[:self.settings.max_audits_per_step]
        if self.settings.frozen or not chosen or self.judge_factory is None:
            return metrics
        judge = self.judge_factory()
        try:
            rules = [] if self.settings.raw_context else self.memory.rules()   # raw mode stores no rules to cover
            results = await asyncio.gather(*[judge.judge(self.persona_for(g), g["dialogues"], rules) for g in chosen])
        finally:
            close = getattr(judge, "aclose", None)
            if close is not None:
                await close()
        collapsed = added = failed = 0
        for g, (parsed, raw) in zip(chosen, results):
            if parsed is not None and self.settings.no_collapse_check:
                parsed["collapsed"] = True   # the LLM judge already forces this; here it also covers the mock judge
            record = {"step": step, "group": g["group"], "task": g["task"], "mean": g["mean"], "std": g["std"],
                      "verdict": parsed, "raw": (raw or "")[:8000]}
            if parsed is None:
                failed += 1
            elif parsed["collapsed"]:
                collapsed += 1
                if self.settings.raw_context:
                    idx, dialogue = pick_dialogue(g, parsed.get("script", ""))
                    note = self.memory.add_context(dialogue, parsed["cause"], step, g["task"], g["group"],
                                                   parsed.get("script", ""), idx)
                    if note is None:
                        record["duplicate_rule"] = True
                    else:
                        record["note_id"], record["context_dialogue"] = note["id"], idx
                        added += 1
                        self._log("context_added", step=step, id=note["id"], task=g["task"], cause=note["cause"],
                                  dialogue=idx, turns=len(note["transcript"]))
                elif parsed["rule"]:
                    # Parallel judges of one step see the same rule list, so the same rule can come back
                    # several times; Memory.add drops exact duplicates.
                    note = self.memory.add(parsed["rule"], parsed["examples"], parsed["cause"], step, g["task"], g["group"])
                    if note is None:
                        record["duplicate_rule"] = True
                    else:
                        record["note_id"] = note["id"]
                        added += 1
                        self._log("rule_added", step=step, id=note["id"], task=g["task"], cause=note["cause"],
                                  rule=note["rule"], examples=len(note["examples"]))
            self._append("audit_log.jsonl", record)
        metrics.update({"simmem/audits": float(len(chosen)), "simmem/judge_failures": float(failed),
                        "simmem/collapsed": float(collapsed), "simmem/notes_added": float(added),
                        "simmem/notes": float(len(self.memory.notes)), "simmem/update_seconds": time.time() - t0})
        return metrics

    def _append(self, name, record):
        with open(os.path.join(self.settings.dir, name), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _log(self, event, **fields):
        self._append("events.jsonl", {"event": event, "time": time.time(), **fields})
        logger.info("[simmem] %s %s", event, json.dumps(fields, ensure_ascii=False, default=str)[:400])


def make_llm_judge(settings):
    from .client import ChatClient
    client = ChatClient((settings.judge_model or "unused").removeprefix("openai/"), settings.judge_base_url or "",
                        settings.judge_api_key_var, timeout=180., max_retries=1, max_concurrency=settings.concurrency)
    return Judge(client, concurrency=settings.concurrency, no_collapse_check=settings.no_collapse_check)


def get_updater(args, judge_factory=None, corpus_dir=None):
    """Process-wide updater keyed by memory directory (the rollout function is re-entered every step)."""
    settings = SimMemSettings.from_args(args)
    if not settings.enable:
        return None
    if settings.dir not in _UPDATERS:
        factory = None if settings.frozen else (judge_factory or (lambda: make_llm_judge(settings)))
        _UPDATERS[settings.dir] = SimMemUpdater(
            settings, factory, corpus_dir if corpus_dir is not None else getattr(args, "p4g_corpus_path", ""))
    return _UPDATERS[settings.dir]
