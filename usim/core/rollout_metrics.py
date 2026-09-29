"""Shared rollout metrics computation for all training pipelines."""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def compute_rollout_metrics(
    grouped_results: List[List[Any]],
    rollout_id: int,
    prefix: str = "ROLLOUT",
) -> Dict[str, Any]:
    """Compute and log rollout metrics including per-sample debug info.

    Args:
        grouped_results: List of sample groups (each group is n_samples_per_prompt).
        rollout_id: Current rollout/step ID.
        prefix: Logging prefix (e.g. "P4G").

    Returns:
        Dict of metric name -> value for wandb logging.
    """
    all_samples = [s for group in grouped_results for s in group]
    rewards = [s.reward for s in all_samples if s.reward is not None]
    turn_counts = [
        s.metadata.get("turn_count", 0)
        for s in all_samples
        if hasattr(s, "metadata") and s.metadata
    ]
    response_lengths = [s.response_length for s in all_samples]
    truncated = [s.status.value == "truncated" for s in all_samples]
    failed = [s.status.value == "failed" for s in all_samples]

    # Token-level policy entropy, Monte-Carlo estimate: mean(-log p) over the
    # sampled agent tokens (rollout-time log probs, i.e. the temperature-scaled
    # behavior policy); the exact logits-based value is train/entropy_loss.
    sample_entropies = []
    for s in all_samples:
        log_probs = getattr(s, "rollout_log_probs", None)
        if log_probs:
            sample_entropies.append(-float(np.mean(log_probs)))

    # Zero-std groups: groups where all samples have the same reward (no learning signal)
    zero_std_groups = 0
    for group in grouped_results:
        group_rewards = [s.reward for s in group if s.reward is not None]
        if len(group_rewards) >= 2:
            std = float(np.std(group_rewards))
            if std == 0.0:
                zero_std_groups += 1

    num_groups = len(grouped_results)
    zero_std_pct = zero_std_groups / max(num_groups, 1)

    metrics = {
        "rollout/num_samples": len(all_samples),
        "rollout/num_groups": num_groups,
        "rollout/raw_reward/mean": float(np.mean(rewards)) if rewards else 0.0,
        "rollout/raw_reward/std": float(np.std(rewards)) if rewards else 0.0,
        "rollout/raw_reward/min": float(np.min(rewards)) if rewards else 0.0,
        "rollout/raw_reward/max": float(np.max(rewards)) if rewards else 0.0,
        "rollout/policy_entropy/mean": float(np.mean(sample_entropies)) if sample_entropies else 0.0,
        "rollout/policy_entropy/std": float(np.std(sample_entropies)) if sample_entropies else 0.0,
        "rollout/turn_count/mean": float(np.mean(turn_counts)) if turn_counts else 0.0,
        "rollout/turn_count/max": int(np.max(turn_counts)) if turn_counts else 0,
        "rollout/response_len/mean": float(np.mean(response_lengths)) if response_lengths else 0.0,
        "rollout/truncated_ratio": float(np.mean(truncated)) if truncated else 0.0,
        "rollout/failed_ratio": float(np.mean(failed)) if failed else 0.0,
        "rollout/zero_std_group_pct": zero_std_pct,
    }

    # Log first failed response for debugging tool call parsing
    for s in all_samples[:1]:
        meta = s.metadata or {}
        if meta.get("first_failed_response"):
            logger.info(
                f"[{prefix}] FIRST FAILED RESPONSE (sample {s.index}): "
                f"{meta['first_failed_response']!r}"
            )

    logger.info(
        f"[{prefix}] Rollout {rollout_id}: "
        f"{len(all_samples)} samples in {num_groups} groups | "
        f"reward={metrics['rollout/raw_reward/mean']:.3f}+-{metrics['rollout/raw_reward/std']:.3f} "
        f"[{metrics['rollout/raw_reward/min']:.3f}, {metrics['rollout/raw_reward/max']:.3f}] | "
        f"entropy={metrics['rollout/policy_entropy/mean']:.3f}+-{metrics['rollout/policy_entropy/std']:.3f} | "
        f"turns={metrics['rollout/turn_count/mean']:.1f} (max={metrics['rollout/turn_count/max']}) | "
        f"resp_len={metrics['rollout/response_len/mean']:.0f} | "
        f"truncated={metrics['rollout/truncated_ratio']:.1%} | "
        f"failed={metrics['rollout/failed_ratio']:.1%} | "
        f"zero_std_groups={zero_std_pct:.1%} ({zero_std_groups}/{num_groups})"
    )
    # Debug: log first few samples
    for s in all_samples[:3]:
        meta = s.metadata or {}
        logger.info(
            f"[{prefix}] Sample {s.index}: "
            f"reward={s.reward:.3f}, turns={meta.get('turn_count', '?')}, "
            f"tokens={len(s.tokens)}, resp_len={s.response_length}, "
            f"status={s.status}, response={s.response[:100]}..."
        )

    return metrics
