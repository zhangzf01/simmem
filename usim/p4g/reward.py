"""Reward computation for Persuasion for Good.

Extracts donation amount from persuadee messages and normalizes to [0, 1].

Regex patterns ported from persuasion_simulation/src/evaluation/metrics/donation_metrics.py.
"""

import re
from typing import List

from usim.core.types import Message


def extract_donation_amount(messages: List[Message]) -> float:
    """Extract the final donation amount from conversation messages.

    Scans persuadee (role="user") messages for donation amounts using:
    1. Structured format: [DONATE $N] or [GIVE $N]
    2. Fallback regex: natural language donation mentions

    Takes the last matched amount (final decision).

    Args:
        messages: Conversation messages

    Returns:
        Donation amount in dollars (0.0 if none found)
    """
    last_amount = 0.0

    # Pattern 1a: Structured format [DONATE $N] / [GIVE $N] ($ optional). A colon
    # after the keyword ("[DONATE: $1.50]") is a simulator formatting slip, not
    # a different decision, so it is accepted too.
    structured_pattern = re.compile(
        r"\[\s*(?:DONATE|GIVE)\s*:?\s*\$?\s*(\d+\.?\d*)\s*\]", re.IGNORECASE
    )
    # Pattern 1b: Bracketed bare amount like [$1.00] — the simulator often
    # drops the keyword but keeps the marker shape. The $ is required here
    # so stray bracketed numbers don't count.
    bracket_amount_pattern = re.compile(r"\[\s*\$\s*(\d+\.?\d*)\s*\]")

    # Pattern 2: Fallback — natural language donation mentions. Tolerates
    # brackets/short qualifiers between the verb and the amount.
    fallback_pattern = re.compile(
        r"\b(?:donate|donating|give|giving|contribute|contributing|"
        r"chip(?:ping)?\s+in|throw(?:ing)?\s+in|toss(?:ing)?\s+in|"
        r"pitch(?:ing)?\s+in|spare|sparing)"
        r"[\s\[\]]*(?:the whole|all|entire|my|another|maybe|like|about|around)?"
        r"[\s\[\]]*\$?\s*(\d*\.\d{1,2}|\d+)\b",
        re.IGNORECASE,
    )
    # Pattern 2b: "put $1.25 in / toward" — amount sits between verb and particle
    put_in_pattern = re.compile(
        r"\bput(?:ting)?\s+\$\s*(\d+\.?\d*)\s+(?:in|toward|towards)\b", re.IGNORECASE
    )
    # A fallback match directly preceded by a negation ("won't give $1",
    # "not donating 2") is a refusal, not a donation.
    negation_pattern = re.compile(
        r"\b(?:not|n'?t|won'?t|wouldn'?t|can'?t|cannot|couldn'?t|never|"
        r"rather\s+not|no\s+way|don'?t\s+want\s+to)\W{0,15}$",
        re.IGNORECASE,
    )

    for msg in messages:
        if msg.role != "user" or not msg.content:
            continue
        text = msg.content

        # Try structured formats first
        structured_matches = sorted(
            list(structured_pattern.finditer(text))
            + list(bracket_amount_pattern.finditer(text)),
            key=lambda m: m.start(),
        )
        if structured_matches:
            last_amount = float(structured_matches[-1].group(1))
            continue

        # Fallback to natural language, skipping negated mentions
        nl_matches = sorted(
            (
                m
                for m in list(fallback_pattern.finditer(text))
                + list(put_in_pattern.finditer(text))
                if not negation_pattern.search(text, 0, m.start())
            ),
            key=lambda m: m.start(),
        )
        if nl_matches:
            last_amount = float(nl_matches[-1].group(1))

    return last_amount


def compute_p4g_reward(messages: List[Message]) -> float:
    """Compute reward from P4G conversation.

    Reward = donation_amount / 2.0, clamped to [0, 1].

    Args:
        messages: Conversation messages

    Returns:
        Reward in [0.0, 1.0]
    """
    donation = extract_donation_amount(messages)
    return min(donation / 2.0, 1.0)
