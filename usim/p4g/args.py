"""Command-line options shared by ``train.py`` and ``evaluate.py`` (added on top of Slime's own arguments)."""
import argparse


def add_p4g_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """User-simulator, P4G, training-rollout and SimMem options."""
    usim_group = parser.add_argument_group("usim", "User simulator (persuadee) arguments")
    usim_group.add_argument(
        "--trainable-role", type=str, default="agent", choices=["agent"],
        help="Only the persuader (agent) is trained.",
    )
    usim_group.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns (P4G: 10)")
    usim_group.add_argument(
        "--usim-fixed-opponent-model", type=str, default=None,
        help="Persuadee model id(s) behind an OpenAI-compatible endpoint (LiteLLM naming, e.g. openai/qwen2.5-72b). "
             "A comma-separated list splits the prompt groups of a step across several simulators.",
    )
    usim_group.add_argument("--usim-fixed-opponent-base-url", type=str, default="https://api.openai.com/v1",
                            help="Base URL(s) of the persuadee endpoint (comma-separated, aligned with the models)")
    usim_group.add_argument("--usim-fixed-opponent-api-key-var", type=str, default="OPENAI_API_KEY",
                            help="Environment variable(s) holding the persuadee API key")
    usim_group.add_argument("--usim-max-tokens", type=int, default=2048,
                            help="Max new tokens per persuadee reply during training rollouts")
    usim_group.add_argument("--usim-verbalized-sampling", action="store_true", default=False,
                            help="GRPO+VS baseline: the training persuadee verbalizes N candidate replies per turn")
    usim_group.add_argument("--usim-vs-num-samples", type=int, default=5)
    usim_group.add_argument("--usim-vs-method", type=str, default="prob", choices=["prob", "random"])

    from usim.p4g.rollout import add_p4g_arguments
    add_p4g_arguments(parser)

    train_group = parser.add_argument_group("p4g-train", "Multi-turn GRPO training rollout")
    train_group.add_argument("--rollout-concurrency", type=int, default=64,
                             help="Concurrent policy generate calls during a training step")
    train_group.add_argument("--rollout-turn-timeout", type=float, default=180.,
                             help="Seconds allowed for one policy generation or one simulator reply")
    train_group.add_argument("--ray-address", default="local")

    # Cross-step user-simulator memory (off unless --simmem-enable).
    from usim.simmem.config import add_arguments as add_simmem_arguments
    add_simmem_arguments(parser)
    return parser
