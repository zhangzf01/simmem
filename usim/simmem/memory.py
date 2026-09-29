"""The memory: a list of rules ("Don't ..." + in-context examples), persisted as JSON, rendered into the persuadee prompt.

Raw-context ablation: an entry may instead be one flagged conversation quoted verbatim (``kind: context``);
entries without ``kind`` are rules.
"""
import json
import os
import time

HEADER = ("Rules from your earlier conversations in this role. Each names a move that made you give in without "
          "a reason of your own, with what you said then and what you should have said. Keep your persona and "
          "your own position; donating is still fine when you actually want to.")
# Raw-context ablation: the block quotes flagged conversations instead of rules. The header says what the
# entries are, like HEADER does; the judge's one-word cause is stored but, like a rule's cause, not rendered.
RAW_HEADER = ("Earlier conversations of yours in this role in which you gave in to the partner without a reason of "
              "your own, quoted in full. Keep your persona and your own position; donating is still fine when you "
              "actually want to.")
RAW_MAX_CHARS = 3000   # per transcript; a longer one keeps its tail (the collapse is at the end)
RAW_WITH_CAUSE = False  # True would tag each transcript with the judge's cause (not rendered for rules either)


def _kind(note):
    return note.get("kind", "rule")


def trim_transcript(transcript, max_chars=RAW_MAX_CHARS):
    msgs = [{"role": m["role"], "content": str(m.get("content") or "")} for m in transcript]
    total, dropped = sum(len(m["content"]) for m in msgs), 0
    while len(msgs) > 1 and total > max_chars:
        total -= len(msgs.pop(0)["content"])
        dropped += 1
    if dropped:
        msgs.insert(0, {"role": "note", "content": f"[{dropped} earlier turns omitted]"})
    return msgs


class Memory:
    def __init__(self, directory):
        self.dir = directory
        self.path = os.path.join(directory, "memory.json")
        self.notes = []

    @classmethod
    def load(cls, directory, init_path=""):
        os.makedirs(directory, exist_ok=True)
        mem = cls(directory)
        source = mem.path if os.path.exists(mem.path) else (init_path or None)
        if source and os.path.exists(source):
            with open(source) as f:
                data = json.load(f)
            mem.notes = data.get("notes", [])
            if source != mem.path:
                mem.save()
        return mem

    def rules(self):
        return [n["rule"] for n in self.notes if _kind(n) == "rule"]

    def has(self, rule):
        return any(_kind(n) == "rule" and n["rule"].strip().lower() == rule.strip().lower()
                   for n in self.notes)

    def add(self, rule, examples, cause, step, task, group):
        """Append a rule with its examples; an exact-text duplicate rule is not stored again (returns None)."""
        if self.has(rule):
            return None
        record = {"id": self._next_id(), "rule": rule.strip(), "examples": list(examples or []),
                  "cause": (cause or "").strip(), "step": step, "task": str(task), "group": group, "time": time.time()}
        self.notes.append(record)
        self.save()
        return record

    def add_context(self, transcript, cause, step, task, group, script="", dialogue_index=None):
        """Raw-context ablation: store one flagged conversation verbatim; one entry per (step, group)."""
        source = f"{step}:{group}"
        if any(_kind(n) == "context" and n.get("source") == source for n in self.notes):
            return None
        record = {"id": self._next_id(), "kind": "context", "rule": "", "examples": [], "source": source,
                  "transcript": trim_transcript(transcript), "dialogue_index": dialogue_index,
                  "cause": (cause or "").strip(), "script": (script or "").strip(),
                  "step": step, "task": str(task), "group": group, "time": time.time()}
        self.notes.append(record)
        self.save()
        return record

    def _next_id(self):
        return max((n["id"] for n in self.notes), default=0) + 1

    def save(self):
        tmp = self.path + ".tmp"
        data = {"notes": self.notes}
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def render(self, max_notes=12):
        seen, unique = set(), []
        for n in self.notes:
            key = f"context:{n.get('source')}" if _kind(n) == "context" else n["rule"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(n)
        notes = unique[-max_notes:]
        if not notes:
            return ""
        raw = all(_kind(n) == "context" for n in notes)
        lines = ["<experience>", RAW_HEADER if raw else HEADER]
        for i, n in enumerate(notes, 1):
            if _kind(n) == "context":
                tag = f" (cause: {n['cause']})" if RAW_WITH_CAUSE and n.get("cause") else ""
                lines.append(f"{i}. Conversation{tag}")
                for m in n["transcript"]:
                    if m["role"] == "note":
                        lines.append(f"   {m['content']}")
                    else:
                        lines.append(f'   {"Partner" if m["role"] == "assistant" else "You"}: "{m["content"]}"')
                continue
            lines.append(f"{i}. {n['rule']}")
            for e in n.get("examples", []):
                lines.append(f'   Partner: "{e["partner"]}"')
                lines.append(f'   You said: "{e["you_said"]}"')
                lines.append(f'   Better: "{e["better"]}"')
        lines.append("</experience>")
        return "\n".join(lines)
