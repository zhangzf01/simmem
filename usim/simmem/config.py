"""``--simmem-*`` options."""
from dataclasses import dataclass, fields


@dataclass
class SimMemSettings:
    enable: bool = False
    dir: str = ""                  # default <trajectory_output_dir>/sim_memory
    mean_threshold: float = 0.9    # group mean reward gate
    std_threshold: float = 0.1     # population std gate (ddof=0)
    max_audits_per_step: int = 4   # judge calls per step, highest-mean groups first
    max_notes: int = 12            # most recent distinct rules rendered into the prompt
    judge_model: str = None        # OpenAI-compatible model id of the judge
    judge_base_url: str = None
    judge_api_key_var: str = "OPENAI_API_KEY"
    concurrency: int = 4
    init: str = ""                 # memory.json to start from
    frozen: bool = False           # never add notes (fixed-memory control)
    env: str = "p4g"
    # Component ablations (paper, "Component ablations"):
    no_collapse_check: bool = False  # audit: the judge skips the collapse verdict, every gated group is treated as collapsed
    raw_context: bool = False        # instruction generation: store the flagged conversation verbatim, no rule / examples
    raw_max_notes: int = 8           # transcripts rendered in raw_context mode (8 x ~1.7k chars ~ a 12-rule block)

    @property
    def render_notes(self):
        """How many entries the prompt block shows."""
        return self.raw_max_notes if self.raw_context else self.max_notes

    @classmethod
    def from_args(cls, args):
        out = cls(**{f.name: getattr(args, f"simmem_{f.name}", f.default) for f in fields(cls) if f.name != "env"})
        out.judge_api_key_var = out.judge_api_key_var or "OPENAI_API_KEY"
        if not out.dir:
            out.dir = f"{(getattr(args, 'trajectory_output_dir', None) or '.').rstrip('/')}/sim_memory"
        return out

    def validate(self):
        if self.enable and not self.frozen and not self.judge_model:
            raise ValueError("Set --simmem-judge-model (and --simmem-judge-base-url) for the memory judge")
        if self.max_audits_per_step < 0 or self.max_notes < 1:
            raise ValueError("simmem counts out of range")
        if self.env != "p4g":
            raise ValueError(f"simmem has no judge prompt for env {self.env!r}")
        if self.raw_max_notes < 1:
            raise ValueError("simmem raw_max_notes out of range")
        return self


def add_arguments(parser):
    group = parser.add_argument_group("simmem", "cross-step user simulator memory")
    group.add_argument("--simmem-enable", action="store_true", default=False)
    group.add_argument("--simmem-dir", default="", help="default: <trajectory-output-dir>/sim_memory")
    group.add_argument("--simmem-mean-threshold", type=float, default=.9)
    group.add_argument("--simmem-std-threshold", type=float, default=.1)
    group.add_argument("--simmem-max-audits-per-step", type=int, default=4)
    group.add_argument("--simmem-max-notes", type=int, default=12)
    group.add_argument("--simmem-concurrency", type=int, default=4)
    group.add_argument("--simmem-judge-model", default=None,
                       help="judge model id on an OpenAI-compatible endpoint (a leading 'openai/' is stripped)")
    group.add_argument("--simmem-judge-base-url", default=None)
    group.add_argument("--simmem-judge-api-key-var", default="OPENAI_API_KEY")
    group.add_argument("--simmem-init", default="", help="memory.json to start from")
    group.add_argument("--simmem-frozen", action="store_true", default=False, help="never add notes")
    group.add_argument("--simmem-no-collapse-check", action="store_true", default=False,
                       help="ablation: the judge skips the collapse verdict and audits every gated group as collapsed")
    group.add_argument("--simmem-raw-context", action="store_true", default=False,
                       help="ablation: append the flagged conversation verbatim instead of a Don't rule + examples")
    group.add_argument("--simmem-raw-max-notes", type=int, default=8,
                       help="transcripts rendered in --simmem-raw-context mode")
    return parser
