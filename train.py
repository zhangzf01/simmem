"""GRPO (+ SimMem with --simmem-enable) training of a P4G persuader on Slime.

    python train.py <slime model/parallel/optimizer args> --usim-fixed-opponent-model ... [--simmem-enable ...]

See scripts/train_p4g_4b.sh for the exact configuration of the paper runs.
"""
import importlib.util
import logging
import os
from pathlib import Path
import tempfile


def add_train_arguments(parser):
    from usim.p4g.args import add_p4g_train_arguments
    return add_p4g_train_arguments(parser)


def configure(args):
    """Route Slime's hooks to the P4G rollout and fail loudly on settings the rollout does not support."""
    if getattr(args, "trainable_role", "agent") != "agent":
        raise ValueError("Only the persuader (agent) is trained")
    if not getattr(args, "usim_fixed_opponent_model", None):
        raise ValueError("--usim-fixed-opponent-model (the training simulator) is required")
    if args.advantage_estimator != "grpo":
        raise ValueError("The P4G rollout supports GRPO outcome advantages only")
    if not getattr(args, "rewards_normalization", True):
        raise ValueError("GRPO needs group-relative reward normalization")
    if getattr(args, "normalize_advantages", False):
        raise ValueError("Global advantage whitening is not used")
    if getattr(args, "qkv_format", "thd") not in ("thd", "bshd"):
        raise ValueError("Unsupported attention layout")
    if getattr(args, "allgather_cp", False):
        raise ValueError("The rollout uses per-sequence context parallel slicing")
    from usim.simmem.config import SimMemSettings
    SimMemSettings.from_args(args).validate()
    args.max_turns = args.p4g_num_turns
    args.rollout_function_path = "usim.p4g.train_rollout.generate_rollout"
    args.eval_function_path = args.rollout_function_path
    args.custom_reward_post_process_path = "usim.p4g.advantage.post_process_rewards"
    args.data_source_path = "usim.p4g.data_source.get_p4g_data_source"
    return args


def load_slime_trainer():
    """Slime keeps ``train.py`` at the root of its repository, outside the installed package."""
    import slime
    candidates = [os.environ.get("SLIME_DIR", "")]
    candidates += [str(Path(p).parent) for p in getattr(slime, "__path__", [])]
    candidates.append(str(Path(__file__).resolve().parent / "third_party" / "slime"))
    for root in candidates:
        path = Path(root) / "train.py" if root else None
        if path is not None and path.is_file():
            spec = importlib.util.spec_from_file_location("slime_train_entry", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Cannot find slime's train.py; set SLIME_DIR to the slime checkout")


def init_ray(args):
    import ray
    names = {
        "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE",
        "OPENAI_BASE_URL", "OPENAI_API_BASE", "USIM_MAX_CONCURRENCY", "USIM_API_MAX_RETRIES",
        "USIM_API_TIMEOUT", "WANDB_MODE", "TENSORBOARD_DIR", "no_proxy", "NO_PROXY",
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        *args.usim_fixed_opponent_api_key_var.split(","),
    }
    if getattr(args, "simmem_judge_api_key_var", None):
        names.add(args.simmem_judge_api_key_var)
    options = {"address": args.ray_address,
               "runtime_env": {"env_vars": {k: os.environ[k] for k in names if k in os.environ}}}
    if args.ray_address == "local":
        options["_temp_dir"] = tempfile.mkdtemp(prefix="p4g-ray-")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            options["num_gpus"] = len(visible.split(","))
    ray.init(**options)
    return ray


def main():
    from slime.utils.arguments import parse_args
    args = configure(parse_args(add_custom_arguments=add_train_arguments))
    trainer = load_slime_trainer()
    logging.getLogger(__name__).info(
        "P4G GRPO: simulator=%s simmem=%s groups/step=%d rollouts/group=%d", args.usim_fixed_opponent_model,
        bool(getattr(args, "simmem_enable", False)), args.rollout_batch_size, args.n_samples_per_prompt)
    ray = init_ray(args)
    try:
        trainer.train(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
