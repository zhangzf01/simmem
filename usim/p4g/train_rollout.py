"""Slime training rollout for P4G: token-exact multi-turn GRPO groups, with the optional SimMem hook.

One training step (``generate_rollout``):
  1. take ``rollout_batch_size`` tasks from the data source, ``n_samples_per_prompt`` rollouts each;
  2. SimMem (if enabled) renders the current memory once; every persuadee of this step gets that block;
  3. run all dialogues against the frozen simulator, keeping the policy's own token ids / log probs;
  4. SimMem audits the high-reward, low-variance groups and appends rules for the *next* step;
  5. hand the samples to Slime; ``usim.p4g.advantage.post_process_rewards`` normalizes rewards per group.

Evaluation (``evaluation=True``) uses the plain P4G eval path in ``usim.p4g.rollout`` and never sees the memory.
"""
import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field

from usim.core.orchestrator import UserSimOrchestrator
from usim.core.types import TrajectoryStatus, UserSimConfig, get_token_delta

logger = logging.getLogger(__name__)

_PERSONAS = {}


@dataclass
class Frame:
    messages: list
    tokens: list
    masks: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    task: dict = field(default_factory=dict)
    elapsed: int = 0
    reward: float = 0.
    info: dict = field(default_factory=dict)
    status: TrajectoryStatus = TrajectoryStatus.TRUNCATED


class Runner(UserSimOrchestrator):
    """Runs one dialogue. The agent's tokens come straight from SGLang; simulator turns are
    tokenized as chat-template deltas with loss mask 0."""

    def __init__(self, tokenizer, config, timeout):
        super().__init__(tokenizer, config)
        self.timeout = timeout

    def append_feedback(self, frame, env, obs):
        frame.messages.append({"role": "user", "content": obs})
        delta, mask = get_token_delta(self.tokenizer, frame.messages,
                                      postprocess=env.prompt_postprocess_fn or (lambda x: x))
        remaining = max(0, self.config.max_context_length - len(frame.tokens))
        delta, mask = delta[:remaining], mask[:remaining]
        frame.tokens.extend(delta)
        frame.masks.extend(mask)
        frame.logprobs.extend([0.] * len(delta))

    async def run(self, env, generate_fn, sampling):
        messages, tools, task = await asyncio.wait_for(env.reset(), self.timeout)
        frame = Frame(list(messages), self._tokenize_prompt(messages, tools, env.prompt_postprocess_fn or (lambda x: x)),
                      task=task)
        try:
            while frame.elapsed < self.config.max_turns:
                if len(frame.tokens) >= self.config.max_context_length:
                    break
                params = dict(sampling)
                # Bound each response by the remaining total token budget.
                params["max_new_tokens"] = min(params.get("max_new_tokens", self.config.max_tokens),
                                               self.config.max_context_length - len(frame.tokens))
                output = await asyncio.wait_for(generate_fn(frame.tokens, params), self.timeout)
                if output.get("error"):
                    raise RuntimeError("Agent inference failed")
                reason = output.get("meta_info", {}).get("finish_reason", {})
                if isinstance(reason, dict) and reason.get("type") == "abort":
                    raise RuntimeError("Agent inference aborted")
                ids, logprobs = output["token_ids"], output["logprobs"]
                if not ids or len(ids) != len(logprobs):
                    raise ValueError("Agent tokens/logprobs are empty or misaligned")
                text = output["text"]
                parsed = env.parse_response(text)
                if parsed is None or parsed["calls"]:
                    raise ValueError("Agent action is malformed")
                frame.messages.append({"role": "assistant", "content": text})
                frame.tokens.extend(ids)
                frame.masks.extend([1] * len(ids))
                frame.logprobs.extend(logprobs)
                frame.elapsed += 1
                if ids[-1] != self._eos_token_id:
                    break   # hit max_new_tokens: the reply is incomplete, stop here
                glue = self._inter_message_glue[:max(0, self.config.max_context_length - len(frame.tokens))]
                frame.tokens.extend(glue)
                frame.masks.extend([0] * len(glue))
                frame.logprobs.extend([0.] * len(glue))
                obs, reward, terminated, truncated, info = await asyncio.wait_for(
                    env.step(parsed["normal_text"]), self.timeout)
                frame.reward, frame.info = float(reward), info
                if not (terminated and info.get("agent_exit")):
                    self.append_feedback(frame, env, obs)
                if terminated or truncated:
                    frame.status = TrajectoryStatus.COMPLETED if terminated else TrajectoryStatus.TRUNCATED
                    break
        except Exception as exc:
            logger.warning("P4G trajectory failed: %s: %s", type(exc).__name__, exc)
            frame.status, frame.reward = TrajectoryStatus.FAILED, 0.
            frame.info = {"error": type(exc).__name__}
            frame.masks = [0] * len(frame.masks)
        if not frame.masks:
            # Slime requires a nonempty response even for a removed sample.
            frame.tokens.append(self._eos_token_id)
            frame.masks, frame.logprobs = [0], [0.]
        return self._build_trajectory(
            frame.tokens, frame.masks, frame.logprobs, frame.messages, frame.elapsed,
            frame.task, frame.status, frame.reward, {**frame.info, "elapsed_agent_turns": frame.elapsed},
        )


def simulator_config(args, sample):
    """(model, base url, api-key variable) of the training simulator for this sample's group.

    ``--usim-fixed-opponent-*`` accept comma-separated lists (the GRPO-ensemble baseline). The bucket is the
    group, so one GRPO baseline never mixes simulators."""
    meta = sample.metadata or {}
    models = meta.get("usim_fixed_opponent_model") or args.usim_fixed_opponent_model
    models = [m.strip() for m in models.split(",") if m.strip()]
    bucket = getattr(sample, "group_index", None)
    if bucket is None:
        bucket = sample.index
    index = (bucket or 0) % len(models)
    urls = meta.get("usim_fixed_opponent_base_url", args.usim_fixed_opponent_base_url).split(",")
    keys = meta.get("usim_fixed_opponent_api_key_var", args.usim_fixed_opponent_api_key_var).split(",")
    return models[index], urls[index % len(urls)].strip(), keys[index % len(keys)].strip()


def environment_factory(args, sample, sim_config, persuadee_memory=""):
    from usim.core.environment.p4g.environment import P4gEnvironment
    from usim.p4g.persona import PersonaLoader
    model, url, key = sim_config
    meta = sample.metadata or {}
    corpus = meta.get("corpus_path", args.p4g_corpus_path)
    if corpus not in _PERSONAS:
        _PERSONAS[corpus] = PersonaLoader(corpus)
    loader = _PERSONAS[corpus]
    persuader = loader.get_persona_text(meta["persuader_speaker_id"])
    persuadee = loader.get_persona_text(meta["persuadee_speaker_id"])
    prefix = meta.get("persuadee_prompt_prefix", getattr(args, "p4g_persuadee_prompt_prefix", "")) or ""
    if isinstance(prefix, list):
        prefix = " ".join(prefix)
    return lambda: P4gEnvironment(
        persuader_persona=persuader, persuadee_persona=persuadee,
        num_turns=args.p4g_num_turns, word_limit=args.p4g_word_limit,
        termination_mode=meta.get("p4g_termination_mode", getattr(args, "p4g_termination_mode", "eager")),
        persuadee_model=model, persuadee_base_url=url, persuadee_api_key=os.getenv(key),
        persuadee_max_tokens=getattr(args, "usim_max_tokens", 2048),
        conversation_id=meta.get("conversation_id", str(sample.index)), persuadee_prompt_prefix=prefix,
        persuadee_memory=persuadee_memory,
        verbalized_sampling=bool(getattr(args, "usim_verbalized_sampling", False)),
        vs_num_samples=int(getattr(args, "usim_vs_num_samples", 5)),
        vs_method=str(getattr(args, "usim_vs_method", "prob")),
    )


def simmem_updater(args, groups):
    """SimMem updater of this run (None unless --simmem-enable)."""
    if not getattr(args, "simmem_enable", False) or not groups:
        return None
    from usim.simmem import get_updater
    corpus = (groups[0][0].metadata or {}).get("corpus_path", args.p4g_corpus_path)
    return get_updater(args, corpus_dir=corpus)


async def run_batch(args, groups, rollout_id, generate=None):
    """Roll out every group of one step. ``generate`` overrides the SGLang call (tests)."""
    from slime.utils.processing_utils import load_tokenizer
    from usim.slime.trajectory_converter import trajectory_to_slime_sample
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if getattr(args, "p4g_disable_thinking", False):
        from usim.p4g.rollout import _disable_thinking
        _disable_thinking(tokenizer)
    config = UserSimConfig(max_turns=args.p4g_num_turns // 2,
                           max_tokens=getattr(args, "p4g_max_tokens", 0) or 8192,
                           max_context_length=args.rollout_max_response_len)
    runner = Runner(tokenizer, config, getattr(args, "rollout_turn_timeout", 180.))
    sem = asyncio.Semaphore(getattr(args, "rollout_concurrency", 64))

    async def sglang_generate(ids, params):
        from slime.utils.http_utils import post
        async with sem:
            output = await post(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", {
                "input_ids": ids, "sampling_params": params, "return_logprob": True,
            })
        meta = output.get("meta_info", {})
        logprobs = meta.get("output_token_logprobs", [])
        return {"text": output["text"], "token_ids": [p[1] for p in logprobs],
                "logprobs": [p[0] for p in logprobs], "meta_info": meta}

    generate = generate or sglang_generate
    sampling = {"temperature": args.rollout_temperature, "top_p": getattr(args, "rollout_top_p", .95),
                "top_k": getattr(args, "rollout_top_k", -1), "max_new_tokens": config.max_tokens}
    # SimMem: one memory snapshot for the whole step; the update after the step is live from the next step.
    updater, memory_block = simmem_updater(args, groups), ""
    if updater is not None:
        memory_block = updater.snapshot(rollout_id)

    async def one(group_index, samples):
        sim_config = simulator_config(args, samples[0])
        make_env = environment_factory(args, samples[0], sim_config, memory_block)
        results = await asyncio.gather(*[runner.run(make_env(), generate, sampling) for _ in samples])
        source_meta = samples[0].metadata or {}
        outputs = []
        for source, tr in zip(samples, results):
            out = trajectory_to_slime_sample(tr, source.index)
            out.group_index = source.group_index
            out.remove_sample = not any(out.loss_mask or [])
            out.metadata["group"] = f"{rollout_id}:{group_index}"
            out.metadata["user_model"] = sim_config[0]
            for key in ("persuader_speaker_id", "persuadee_speaker_id"):
                if key in source_meta:
                    out.metadata[key] = source_meta[key]
            if memory_block:
                out.metadata["simmem_notes"] = len(updater.memory.notes)
            outputs.append(out)
        return outputs

    outputs = await asyncio.gather(*[one(i, group) for i, group in enumerate(groups)])
    metrics = {}
    if updater is not None:
        rows = [{"task": s.metadata.get("task_id"), "group": s.metadata["group"], "reward": float(s.reward or 0.),
                 "messages": s.metadata.get("messages", []), "persona_id": s.metadata.get("persuadee_speaker_id")}
                for group in outputs for s in group]
        metrics.update(await updater.after_step(rollout_id, rows))
    return outputs, metrics


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        from usim.p4g.rollout import _p4g_eval_rollout
        return _p4g_eval_rollout(args, rollout_id, data_source)
    from slime.rollout.base_types import RolloutFnTrainOutput
    from usim.core.rollout_metrics import compute_rollout_metrics
    from usim.core.trajectory_recorder import TrajectoryRecorder
    samples, metrics = asyncio.run(run_batch(args, data_source.get_samples(args.rollout_batch_size), rollout_id))
    flat = [sample for group in samples for sample in group]
    groups = defaultdict(list)
    for sample in flat:
        groups[sample.metadata["group"]].append(sample)
    metrics.update(compute_rollout_metrics(list(groups.values()), rollout_id, prefix="P4G"))
    TrajectoryRecorder(getattr(args, "trajectory_output_dir", None)).record_batch(flat, rollout_id)
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
