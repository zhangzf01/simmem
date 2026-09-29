"""P4G evaluation rollout (held-out simulators, one dialogue per test task) and the P4G command-line options.

Training uses ``usim.p4g.train_rollout``; its ``evaluation=True`` branch lands in ``_p4g_eval_rollout`` below.
Evaluation environments are built without the SimMem memory.
"""

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any, Callable, Dict, List

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from usim.core.environment.p4g import P4gEnvironment
from usim.core.orchestrator import UserSimOrchestrator
from usim.core.trajectory_recorder import TrajectoryRecorder, eval_trajectory_dir
from usim.core.types import TrainableRole, UserSimConfig
from usim.p4g.persona import PersonaLoader
from usim.slime.trajectory_converter import failed_placeholder_sample, trajectory_to_slime_sample


logger = logging.getLogger(__name__)

_persona_loader_cache: Dict[str, PersonaLoader] = {}


def _get_persona_loader(corpus_path: str) -> PersonaLoader:
    if corpus_path not in _persona_loader_cache:
        _persona_loader_cache[corpus_path] = PersonaLoader(corpus_path)
    return _persona_loader_cache[corpus_path]


def _get_persuadee_config(args: Any, sample: Any) -> tuple:
    metadata = getattr(sample, "metadata", {}) or {}

    fixed_model = metadata.get("usim_fixed_opponent_model")
    base_url = metadata.get(
        "usim_fixed_opponent_base_url",
        getattr(args, "usim_fixed_opponent_base_url", "https://api.openai.com/v1"),
    )
    api_key_var = metadata.get(
        "usim_fixed_opponent_api_key_var",
        getattr(args, "usim_fixed_opponent_api_key_var", "OPENAI_API_KEY"),
    )

    if not fixed_model:
        fixed_model = getattr(args, "usim_fixed_opponent_model", None)

    if not fixed_model:
        return None, base_url, None

    model_list = [m.strip() for m in fixed_model.split(",") if m.strip()]
    model_idx = sample.index % len(model_list)
    model_name = model_list[model_idx]

    base_url_list = [u.strip() for u in base_url.split(",") if u.strip()]
    api_key_var_list = [k.strip() for k in api_key_var.split(",") if k.strip()]
    model_base_url = base_url_list[model_idx % len(base_url_list)]
    model_api_key_var = api_key_var_list[model_idx % len(api_key_var_list)]
    api_key = os.environ.get(model_api_key_var)

    return model_name, model_base_url, api_key



def _disable_thinking(tokenizer: Any) -> None:
    """Force ``enable_thinking=False`` on every chat-template call.

    Qwen3 base models reason inside ``<think>`` by default; on P4G that burns the
    whole per-turn token budget before the persuader says anything, so every reply
    gets truncated and the reward stays near zero. The flag only affects rollouts
    launched with ``--p4g-disable-thinking`` (the Qwen3-8B arm).
    """
    original = getattr(tokenizer, "apply_chat_template", None)
    if original is None:
        return

    def patched(conversation, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        try:
            return original(conversation, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return original(conversation, **kwargs)

    tokenizer.apply_chat_template = patched


async def _p4g_generate_single(
    args: Any,
    sample: Any,
    sampling_params: Dict[str, Any],
    **kwargs: Any,
) -> Sample:
    """Per-sample async generate function for P4G."""
    try:
        metadata = getattr(sample, "metadata", {}) or {}

        num_turns = getattr(args, "p4g_num_turns", 10)
        word_limit = getattr(args, "p4g_word_limit", 50)
        num_exchanges = num_turns // 2

        corpus_path = metadata.get(
            "corpus_path", getattr(args, "p4g_corpus_path", "")
        )
        persuader_speaker_id = metadata["persuader_speaker_id"]
        persuadee_speaker_id = metadata["persuadee_speaker_id"]

        persona_loader = _get_persona_loader(corpus_path)
        persuader_persona = persona_loader.get_persona_text(persuader_speaker_id)
        persuadee_persona = persona_loader.get_persona_text(persuadee_speaker_id)

        persuadee_model, persuadee_base_url, persuadee_api_key = (
            _get_persuadee_config(args, sample)
        )
        if not persuadee_model:
            raise ValueError(
                "P4G requires --usim-fixed-opponent-model for the persuadee"
            )

        conversation_id = metadata.get("conversation_id", str(sample.index))
        # persuadee prompt prefix: eval injects it via metadata (see eval config
        # metadata_overrides) so it does NOT touch training rollouts, which read
        # only the (empty) args value. metadata wins when present.
        meta_prefix = metadata.get("persuadee_prompt_prefix")
        if meta_prefix is not None:
            persuadee_prompt_prefix = (
                " ".join(meta_prefix) if isinstance(meta_prefix, list) else str(meta_prefix)
            )
        else:
            raw_prefix = getattr(args, "p4g_persuadee_prompt_prefix", []) or []
            persuadee_prompt_prefix = " ".join(raw_prefix) if isinstance(raw_prefix, list) else (raw_prefix or "")
        verbalized_sampling = bool(getattr(args, "usim_verbalized_sampling", False))
        vs_num_samples = int(getattr(args, "usim_vs_num_samples", 5))
        vs_method = str(getattr(args, "usim_vs_method", "prob"))
        env = P4gEnvironment(
            persuader_persona=persuader_persona,
            persuadee_persona=persuadee_persona,
            word_limit=word_limit,
            num_turns=num_turns,
            token_limit=getattr(args, "p4g_max_tokens", 0),
            termination_mode=metadata.get("p4g_termination_mode", getattr(args, "p4g_termination_mode", "eager")),
            persuadee_model=persuadee_model,
            persuadee_base_url=persuadee_base_url,
            persuadee_api_key=persuadee_api_key,
            conversation_id=conversation_id,
            persuadee_prompt_prefix=persuadee_prompt_prefix,
            verbalized_sampling=verbalized_sampling,
            vs_num_samples=vs_num_samples,
            vs_method=vs_method,
        )

        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )

        async def generate_fn(
            input_ids: list, samp_params: Dict[str, Any]
        ) -> Dict[str, Any]:
            output = await post(sglang_url, {
                "input_ids": input_ids,
                "sampling_params": samp_params,
                "return_logprob": True,
            })
            meta = output.get("meta_info", {})
            token_logprobs = meta.get("output_token_logprobs", [])
            return {
                "text": output["text"],
                "token_ids": [item[1] for item in token_logprobs],
                "logprobs": [item[0] for item in token_logprobs],
                "meta_info": meta,
            }

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        if getattr(args, "p4g_disable_thinking", False):
            _disable_thinking(tokenizer)

        config = UserSimConfig(
            trainable_role=TrainableRole(getattr(args, "trainable_role", "agent")),
            max_turns=num_exchanges,
            max_tokens=getattr(args, "p4g_max_tokens", 0) or getattr(args, "max_tokens", 8192),
            max_context_length=getattr(args, "rollout_max_response_len", 32768),
            temperature=sampling_params.get("temperature", 0.7),
        )

        orchestrator = UserSimOrchestrator(tokenizer=tokenizer, config=config)
        trajectory = await orchestrator.rollout(env, generate_fn, sampling_params)

        logger.info(
            f"P4G rollout {sample.index}: {trajectory.turn_count} turns, "
            f"reward={trajectory.reward:.2f}, status={trajectory.status}"
        )

        out = trajectory_to_slime_sample(trajectory, sample.index, group_index=sample.group_index)
        out.metadata["persuadee_speaker_id"] = persuadee_speaker_id
        return out

    except Exception as e:
        logger.error(f"P4G rollout failed (sample {sample.index}): {e}", exc_info=True)
        # Never hand slime an empty sample (zero tokens / None log probs crash
        # the training step); emit a masked-out placeholder instead.
        return failed_placeholder_sample(sample, e, hf_checkpoint=getattr(args, "hf_checkpoint", None))


def _eval_max_concurrency() -> int:
    """Concurrency cap for persuadee API calls (protects fragile sim endpoints).

    Reads USIM_MAX_CONCURRENCY (default 32). Without a cap, a batch fires all
    episodes at the sim at once, which overwhelms rate-limited / self-hosted APIs.
    """
    try:
        return max(1, int(os.environ.get("USIM_MAX_CONCURRENCY", "32")))
    except (TypeError, ValueError):
        return 32


async def _bounded_generate(sem, args, sample, sampling_params):
    async with sem:
        return await _p4g_generate_single(args, sample, sampling_params)


async def _p4g_eval_single_dataset(
    args: Namespace,
    dataset_cfg: Any,
    data_source: Any,
    rollout_id: int = 0,
) -> dict:
    """Run eval for a single P4G eval dataset config."""
    metadata_overrides = getattr(dataset_cfg, "metadata_overrides", {}) or {}
    n_samples = getattr(dataset_cfg, "n_samples_per_eval_prompt", 1)
    sampling_params = dict(
        temperature=getattr(dataset_cfg, "temperature", 0.7),
        top_p=getattr(dataset_cfg, "top_p", 0.95),
        top_k=getattr(dataset_cfg, "top_k", -1),
        max_new_tokens=getattr(args, "p4g_max_tokens", 0)
            or getattr(dataset_cfg, "max_response_len", None)
            or args.rollout_max_response_len,
    )

    # Override user model config from metadata
    eval_args = Namespace(**vars(args))
    if "usim_fixed_opponent_model" in metadata_overrides:
        eval_args.usim_fixed_opponent_model = metadata_overrides["usim_fixed_opponent_model"]
    if "usim_fixed_opponent_base_url" in metadata_overrides:
        eval_args.usim_fixed_opponent_base_url = metadata_overrides["usim_fixed_opponent_base_url"]
    if "usim_fixed_opponent_api_key_var" in metadata_overrides:
        eval_args.usim_fixed_opponent_api_key_var = metadata_overrides["usim_fixed_opponent_api_key_var"]
    # Verbalized Sampling is a training-simulator intervention. Periodic eval talks to
    # the held-out simulator with the stock persuadee prompt unless the eval config
    # opts in explicitly, so the GRPO and GRPO+VS arms stay comparable.
    eval_args.usim_verbalized_sampling = bool(metadata_overrides.get("usim_verbalized_sampling", False))

    # Get test samples from data_source
    eval_samples = data_source.get_eval_samples() if hasattr(data_source, "get_eval_samples") else data_source.get_samples(16)

    sem = asyncio.Semaphore(_eval_max_concurrency())
    coros = []
    sample_idx = 0
    for group in eval_samples:
        for sample in group:
            for j in range(n_samples):
                s = Sample(
                    index=sample_idx,
                    prompt=sample.prompt,
                    metadata={**(sample.metadata or {}), **metadata_overrides},
                )
                sample_idx += 1
                coros.append(_bounded_generate(sem, eval_args, s, sampling_params))

    results = await asyncio.gather(*coros, return_exceptions=True)

    data = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"P4G eval sample failed: {r}")
            data.append(Sample(index=0, reward=0.0))
        else:
            data.append(r)

    # Record eval trajectories: one directory per dataset, file named by the
    # real rollout_id, so periodic evals and concurrent datasets never
    # overwrite each other.
    output_dir = getattr(args, "trajectory_output_dir", None)
    if output_dir:
        recorder = TrajectoryRecorder(eval_trajectory_dir(output_dir, dataset_cfg.name))
        recorder.record_batch(
            data, rollout_id=rollout_id, extra_metadata={"eval_dataset": dataset_cfg.name}
        )

    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None)
    return {
        dataset_cfg.name: {
            "rewards": [
                s.reward if not reward_key else s.reward[reward_key]
                for s in data
            ],
            "truncated": [s.status == Sample.Status.TRUNCATED for s in data],
            "samples": data,
        }
    }


def _p4g_eval_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
) -> RolloutFnEvalOutput:
    """Run P4G eval across all configured eval datasets."""
    eval_datasets = getattr(args, "eval_datasets", []) or []
    if not eval_datasets:
        logger.warning("[P4G] No eval datasets configured")
        return RolloutFnEvalOutput(data={})

    async def _run_all():
        coros = [_p4g_eval_single_dataset(args, cfg, data_source, rollout_id) for cfg in eval_datasets]
        results_list = await asyncio.gather(*coros)
        results = {}
        for r in results_list:
            results.update(r)
        return results

    data = asyncio.run(_run_all())
    return RolloutFnEvalOutput(data=data)


def add_p4g_arguments(parser: Any) -> None:
    """Add P4G-specific arguments to argparse parser."""
    group = parser.add_argument_group("p4g", "Persuasion for Good arguments")
    group.add_argument(
        "--p4g-termination-mode", choices=["eager", "wait"], default="eager",
        help="eager: end on donation; wait: continue until agent [exit] or turn limit. Shared by train/eval.",
    )

    group.add_argument(
        "--p4g-corpus-path",
        type=str,
        default="data/p4g/corpus",
        help="Path to convokit Corpus for persona loading",
    )

    group.add_argument(
        "--p4g-dataset-dir",
        type=str,
        default="data/p4g/train",
        help="Path to directory with dialogue JSONL files",
    )

    group.add_argument(
        "--p4g-eval-dataset-dir",
        type=str,
        default=None,
        help="Path to held-out eval dialogue JSONL files "
        "(default: sibling 'test' dir of --p4g-dataset-dir)",
    )

    group.add_argument(
        "--p4g-word-limit",
        type=int,
        default=50,
        help="Max words per response in prompts (default: 50)",
    )

    group.add_argument(
        "--p4g-num-turns",
        type=int,
        default=10,
        help="Total conversation turns, each side gets num_turns//2 (default: 10)",
    )

    group.add_argument(
        "--p4g-max-tokens",
        type=int,
        default=0,
        help="Hard cap on generated tokens per turn (0 = keep the legacy 8192 default). "
        "Set ~2.5x --p4g-word-limit to enforce the word limit on models that ignore it.",
    )

    group.add_argument(
        "--p4g-disable-thinking",
        action="store_true",
        default=False,
        help="Force enable_thinking=False when applying the chat template. Needed for "
        "Qwen3 base models, which otherwise spend the whole token budget on <think> "
        "reasoning and never reach the actual reply. Off by default.",
    )

    group.add_argument(
        "--p4g-persuadee-prompt-prefix",
        nargs="*",
        default=[],
        help="Prefix prepended to persuadee system prompt (for RL-Configured baseline). "
        "Uses nargs=* to survive unquoted expansion through ray job submit's /bin/sh.",
    )

    group.add_argument(
        "--trajectory-output-dir",
        type=str,
        default=None,
        help="Directory to save trajectory JSONL files (disabled if not set)",
    )
    return parser
