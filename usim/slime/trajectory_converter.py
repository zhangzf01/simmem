"""Convert usim Trajectory to Slime Sample format."""

import logging
from typing import Any, Dict, List, Optional

from usim.core.types import Trajectory, TrajectoryStatus


logger = logging.getLogger(__name__)


def trajectory_to_slime_sample(
    trajectory: Trajectory,
    index: int = 0,
    role: str = "actor",
    group_index: Optional[int] = None,
) -> Any:
    """Convert a usim Trajectory to Slime's Sample format.

    This is the bridge between usim's framework-agnostic Trajectory
    and Slime's training data format.

    Args:
        trajectory: usim Trajectory to convert
        index: Sample index for batching
        role: Role identifier ("actor" or "environment")
        group_index: GRPO group id of the source sample. Slime's own
            per-group metrics (``zero_std/*``) group by this field, so
            callers should pass ``sample.group_index`` through.

    Returns:
        Slime Sample object
    """
    try:
        from slime.utils.types import Sample
    except ImportError:
        raise ImportError("slime package required for trajectory conversion")

    # Map TrajectoryStatus to Slime Sample.Status
    status_map = {
        TrajectoryStatus.COMPLETED: Sample.Status.COMPLETED,
        TrajectoryStatus.TRUNCATED: Sample.Status.TRUNCATED,
        TrajectoryStatus.ABORTED: Sample.Status.ABORTED,
        TrajectoryStatus.FAILED: Sample.Status.FAILED,
        TrajectoryStatus.TIMEOUT: Sample.Status.TRUNCATED,
        TrajectoryStatus.PENDING: Sample.Status.PENDING,
        TrajectoryStatus.RUNNING: Sample.Status.PENDING,
    }
    slime_status = status_map.get(trajectory.status, Sample.Status.PENDING)

    # Build metadata, ensuring sample_weight is preserved for loss weighting
    metadata = {
        "role": role,
        "turn_count": trajectory.turn_count,
        "sample_weight": trajectory.metadata.get("sample_weight", 1.0),
        "messages": trajectory.messages,
        **trajectory.metadata,
    }

    # Determine response_length from loss_mask if not set
    response_length = trajectory.response_length
    if response_length == 0 and trajectory.loss_mask:
        response_length = sum(trajectory.loss_mask)

    return Sample(
        index=index,
        group_index=group_index,
        prompt=trajectory.prompt,
        tokens=trajectory.tokens,
        response=trajectory.response,
        response_length=response_length,
        reward=trajectory.reward,
        loss_mask=trajectory.loss_mask,
        rollout_log_probs=(
            trajectory.rollout_log_probs if trajectory.rollout_log_probs else None
        ),
        status=slime_status,
        metadata=metadata,
    )


def failed_placeholder_sample(
    source: Any,
    error: Any,
    tokenizer: Any = None,
    hf_checkpoint: Optional[str] = None,
) -> Any:
    """Training-safe stand-in for a rollout that raised before producing a trajectory.

    Slime cannot ingest an empty sample: zero tokens break the context-parallel
    slicing, and ``rollout_log_probs=None`` breaks the actor-side conversion as
    soon as any other sample in the batch carries log probs. Mirror what the
    training runner does for a failed trajectory instead: a short prompt plus one
    EOS response token, an all-zero loss mask and ``remove_sample=True`` so the
    sample never contributes to the loss. ``status`` is FAILED so rollout
    metrics count it, and ``metadata["error"]`` keeps the reason.

    Args:
        source: The input Sample the rollout was started from (index,
            group_index, prompt and scalar metadata are carried over).
        error: Exception or message stored under ``metadata["error"]``.
        tokenizer: Optional tokenizer used to build a real prompt prefix.
        hf_checkpoint: Loaded lazily when ``tokenizer`` is None. If neither
            is available the placeholder falls back to token id 0.
    """
    from slime.utils.types import Sample

    if tokenizer is None and hf_checkpoint:
        try:
            from slime.utils.processing_utils import load_tokenizer

            tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed_placeholder_sample: tokenizer unavailable ({e}); using token id 0")

    eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    if eos is None:
        eos = 0
    prompt_tokens: List[int] = []
    if tokenizer is not None:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": "rollout failed"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
        except Exception:  # noqa: BLE001
            prompt_tokens = []
    if not prompt_tokens:
        prompt_tokens = [eos]

    source_meta = getattr(source, "metadata", None) or {}
    scalar_meta = {k: v for k, v in source_meta.items() if isinstance(v, (str, int, float, bool))}
    return Sample(
        index=getattr(source, "index", 0),
        group_index=getattr(source, "group_index", None),
        prompt=getattr(source, "prompt", ""),
        tokens=prompt_tokens + [eos],
        response="",
        response_length=1,
        reward=0.0,
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.FAILED,
        remove_sample=True,
        metadata={
            **scalar_meta,
            "role": "actor",
            "turn_count": 0,
            "messages": [],
            "error": str(error),
        },
    )
