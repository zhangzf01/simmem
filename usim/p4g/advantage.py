"""Group-relative GRPO rewards, grouped by explicit group ids (Slime's ``--custom-reward-post-process-path`` hook)."""
from collections import defaultdict
import math


def _normalize(values, use_std=True):
    if not values:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.
    return [(v - mean) / (std + 1e-6) if use_std else v - mean for v in values]


def post_process_rewards(args, samples):
    """Returns (raw rewards, normalized rewards).

    The group of a sample is ``metadata["group"]`` (one task, one step). Samples whose rollout failed
    (``remove_sample`` / empty loss mask) are left out of the group statistics and get reward 0.
    The std uses ddof=1, as in Slime's stock GRPO normalization.
    """
    groups = defaultdict(list)
    raw = [float(sample.get_reward_value(args)) for sample in samples]
    rewards = [0.] * len(samples)
    for i, sample in enumerate(samples):
        group = (sample.metadata or {}).get("group")
        if group is None:
            raise ValueError("Missing group metadata; was usim.p4g.train_rollout.generate_rollout used?")
        if sample.remove_sample or not any(sample.loss_mask or []):
            continue
        groups[group].append(i)
    for indices in groups.values():
        normalized = _normalize([raw[i] for i in indices], getattr(args, "grpo_std_normalization", True))
        for i, value in zip(indices, normalized):
            rewards[i] = value
    return raw, rewards
