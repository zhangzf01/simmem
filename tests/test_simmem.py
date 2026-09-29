"""CPU tests for SimMem (no GPU, no Slime, no API).  Run:  python -m pytest tests -q

The last two tests replay the logs of the paper's Qwen3-4B SimMem run (results/qwen3_4b_simmem/sim_memory):
the memory renderer must reproduce the size of every one of the 250 recorded prompt blocks, and the judge-output
parser must reproduce every recorded verdict.
"""
import asyncio
import json
import os

from usim.simmem.config import SimMemSettings
from usim.simmem.judge import SYSTEM, SYSTEM_NOCHECK, MockJudge, build_messages, parse
from usim.simmem.memory import HEADER, Memory
from usim.simmem.updater import SimMemUpdater, suspicious_groups

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "results", "qwen3_4b_simmem", "sim_memory")


def _row(group, reward, text="hi", donate=None):
    user = f"{text} [DONATE ${donate}]" if donate is not None else text
    return {"task": group.split(":")[1], "group": group, "reward": reward, "persona_id": None,
            "messages": [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "ask"},
                         {"role": "user", "content": user}]}


def test_gate_uses_mean_and_population_std():
    s = SimMemSettings(enable=True, frozen=True)
    rows = ([_row("0:0", 1.) for _ in range(8)]                                  # mean 1, std 0      -> in
            + [_row("0:1", r) for r in [.75] + [1.] * 7]                       # mean .969, std .083 -> in
            + [_row("0:2", r) for r in [.5, .5] + [1.] * 6]                    # std .217           -> out
            + [_row("0:3", .8) for _ in range(8)])                             # mean .8            -> out
    groups, n = suspicious_groups(rows, s)
    assert n == 4
    assert [g["group"] for g in groups] == ["0:0", "0:1"]   # highest mean first
    assert abs(groups[1]["std"] - 0.0827) < 1e-3            # ddof=0


def test_judge_prompt_and_payload():
    msgs = build_messages({"age": 30}, [[{"role": "assistant", "content": "a"}]], ["Don't x."])
    assert msgs[0]["content"] == SYSTEM
    payload = json.loads(msgs[1]["content"])
    assert payload == {"persona": {"age": 30}, "conversations": [{"id": "C0", "messages": [
        {"role": "assistant", "content": "a"}]}], "existing_rules": ["Don't x."]}
    assert "Step 1 - describe the script" in SYSTEM_NOCHECK and "Step 1 - has the group collapsed?" not in SYSTEM_NOCHECK


def test_parse_filters_examples_with_amounts():
    text = "```json\n" + json.dumps({"collapsed": True, "script": "s", "cause": "misreading", "rule": "Don't x.",
                                     "examples": [{"partner": "p", "you_said": "y", "better": "Show me a report."},
                                                  {"partner": "p", "you_said": "y", "better": "Maybe $1."},
                                                  {"partner": "p", "you_said": "", "better": "b"}]}) + "\n```"
    out = parse(text)
    assert out["collapsed"] and out["rule"] == "Don't x."
    assert out["examples"] == [{"partner": "p", "you_said": "y", "better": "Show me a report."}]
    assert parse("no json here") is None


def test_memory_dedup_and_render(tmp_path):
    mem = Memory.load(str(tmp_path))
    assert mem.render() == ""
    assert mem.add("Don't a.", [{"partner": "P", "you_said": "Y", "better": "B"}], "sycophancy", 1, "7", "1:0")
    assert mem.add("  don't A.  ", [], "", 2, "7", "2:0") is None            # exact text duplicate
    for i in range(13):
        mem.add(f"Don't b{i}.", [], "", 3, "7", "3:0")
    block = mem.render(12)
    assert block.startswith("<experience>\n" + HEADER) and block.endswith("</experience>")
    assert "Don't a." not in block and "12. Don't b12." in block                # the 12 most recent rules
    assert len(Memory.load(str(tmp_path)).notes) == 14                          # persisted, nothing dropped


def test_updater_with_mock_judge(tmp_path):
    s = SimMemSettings(enable=True, dir=str(tmp_path), judge_model="mock")
    up = SimMemUpdater(s, MockJudge)
    assert up.snapshot(0) == ""
    # "budget" is a concern cue; the partner repeats the ask and the persuadee raises its amount.
    dialogue = [{"role": "assistant", "content": "Donate?"},
                {"role": "user", "content": "I have a tight budget. [DONATE $0.5]"},
                {"role": "assistant", "content": "Every dollar counts!"},
                {"role": "user", "content": "Okay. [DONATE $2]"}]
    rows = [{"task": "5", "group": "0:0", "reward": 1., "persona_id": None, "messages": dialogue} for _ in range(8)]
    metrics = asyncio.run(up.after_step(0, rows))
    assert metrics["simmem/audits"] == 1 and metrics["simmem/notes_added"] == 1
    assert "1. Don't raise your amount" in up.snapshot(1)
    metrics = asyncio.run(up.after_step(1, rows))                               # covered now: no new rule
    assert metrics["simmem/notes_added"] == 0 and len(up.memory.notes) == 1


def test_render_reproduces_paper_run_snapshots(tmp_path):
    notes = json.load(open(os.path.join(RUN, "memory.json")))["notes"]
    events = [json.loads(line) for line in open(os.path.join(RUN, "events.jsonl"))]
    snaps = {e["step"]: e for e in events if e["event"] == "snapshot"}
    assert len(snaps) == 250 and len(notes) == 12
    mem = Memory(str(tmp_path))
    for step, e in snaps.items():
        mem.notes = [n for n in notes if n["step"] < step]   # a rule written after step t is live from t + 1
        assert len(mem.render(12)) == e["chars"], step


def test_parser_reproduces_paper_run_verdicts():
    records = [json.loads(line) for line in open(os.path.join(RUN, "audit_log.jsonl"))]
    assert len(records) == 84
    for r in records:
        assert parse(r["raw"]) == r["verdict"], r["step"]
