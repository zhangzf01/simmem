# SimMem: online simulator memory for multi-turn dialogue RL

Code for the ICLR submission *Mitigating the Preference Trap of LLM User Simulators in Non-Collaborative Dialogue RL*.
This repository reproduces the **Qwen3-4B-Instruct-2507** results on Persuasion for Good (P4G): the SimMem arm, the
single-simulator GRPO baseline it is compared with (the same code with SimMem switched off), the held-out evaluation
of Table 1, and the component ablations of the 4B policy.

SimMem keeps a text memory that is appended to the frozen training simulator's system prompt. After every GRPO step
it (1) selects rollout groups with high mean reward and low reward variance, (2) asks a judge whether the group has
collapsed onto one script and which simulator concession makes it work, and (3) writes a `Don't ...` rule with up to
two in-context examples. The rules are live from the next step. Policy, simulator weights, reward and RL algorithm
are unchanged, and evaluation never sees the memory.

## Layout

```
train.py                       entry point: GRPO, plus SimMem with --simmem-enable (also used for evaluation)
usim/simmem/                   SimMem
  updater.py                     per-step hook: snapshot before the step, gate + judge + write after it
  judge.py                       judge prompt (incl. the two worked examples), JSON parsing, example filter
  memory.py                      rule store (memory.json) and the <experience> block rendered into the prompt
  config.py                      --simmem-* options
  offline.py                     replay the memory loop over saved trajectories
usim/p4g/train_rollout.py      one GRPO step: token-exact multi-turn rollouts against the simulator + SimMem hook
usim/p4g/advantage.py          group-relative reward normalization (Slime reward post-process hook)
usim/p4g/rollout.py            held-out evaluation rollout; P4G options
usim/core/                     P4G environment (persuadee calls, donation reward), token bookkeeping
data/p4g/                      739 training / 200 test tasks and the ConvoKit persona corpus
scripts/                       environment setup, simulator server, training, evaluation, scoring
configs/eval/                  held-out evaluation config template
patches/slime.patch            small fixes applied on top of slime@8efb1166
results/qwen3_4b_simmem/       memory, judge audit log and events of the paper's 4B SimMem run
tests/                         CPU tests (include a replay of the paper run's memory logs)
```

The multi-turn rollout code is derived from the SCOPE `usim` package and runs on the
[slime](https://github.com/THUDM/slime) RL framework (Megatron-LM training, SGLang generation).

## 1. Environment

Hardware used in the paper: one node with 8 x NVIDIA H20 (96 GB) per training run. GPUs 0-3 serve the 72B simulator,
GPUs 4-7 train (2 actor GPUs with context parallel 2, 2 SGLang rollout GPUs). A 250-step run takes about 14 hours.

1. Build slime's environment exactly as slime commit `8efb1166` does
   ([`build_conda.sh`](https://github.com/THUDM/slime/blob/8efb1166/build_conda.sh) or slime's Docker image of that
   commit): CUDA 12.9, PyTorch 2.9.1, SGLang, Megatron-LM `3714d81d`, TransformerEngine 2.10, flash-attn 2.7.4.post1.
2. Inside it, from the repository root:
   ```bash
   bash scripts/setup_env.sh      # clones slime@8efb1166 to third_party/slime, applies patches/slime.patch, installs deps
   python -m pytest tests -q      # CPU tests, ~1 s
   ```
   `MEGATRON_ROOT` (default `/root/Megatron-LM`, where slime's build script puts it) must point to the Megatron-LM checkout.

`patches/slime.patch` makes three changes: the actor switches back to its own weights before the first weight push
(matters when evaluating a saved checkpoint with `--load`), micro-batch packing is first-fit-decreasing with no empty
micro-batch, and a timer that was ended before being started no longer raises.

## 2. Models and endpoints

| Role | Model | How |
| --- | --- | --- |
| Policy | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `models/Qwen3-4B-Instruct-2507`, then `bash scripts/convert_hf_to_torch_dist.sh` |
| Training simulator | [Qwen2.5-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ) | `models/Qwen2.5-72B-Instruct-AWQ`, `bash scripts/serve_simulator.sh` (SGLang, GPUs 0-3, port 30010, ~70 s to start) |
| SimMem judge | DeepSeek-V4-Flash, temperature 0 | any OpenAI-compatible endpoint: `JUDGE_BASE_URL`, `JUDGE_MODEL`, key in `$JUDGE_API_KEY` |
| Held-out persuadees | GPT-4o-mini, Doubao-Seed-2.0-mini, Ministral-3-14B-Instruct-2512 | any OpenAI-compatible endpoint (evaluation only) |

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ --local-dir models/Qwen2.5-72B-Instruct-AWQ
bash scripts/convert_hf_to_torch_dist.sh
nohup bash scripts/serve_simulator.sh > simulator.log 2>&1 &
```

The judge is called at most four times per step (after the rollouts, under a minute per step). The persuadee and
the judge are reached through LiteLLM / the OpenAI client, so a model id is written `openai/<id>` for the persuadee.

## 3. Training

```bash
export JUDGE_API_KEY=...                 # key of the judge endpoint
JUDGE_BASE_URL=https://<judge-endpoint>/v1 JUDGE_MODEL=<DeepSeek-V4-Flash id on that endpoint> \
ARM=simmem bash scripts/train_p4g_4b.sh  # GRPO + SimMem

ARM=grpo bash scripts/train_p4g_4b.sh    # GRPO baseline (identical except --simmem-enable)
```

Run a smoke test first (about 10 minutes): `ARM=simmem NUM_ROLLOUT=2 SAVE_INTERVAL=1 bash scripts/train_p4g_4b.sh`.

Configuration (paper Appendix, Table of hyperparameters): 250 steps; 16 tasks per step (training tasks are cycled in
order) x 8 rollouts per task; GRPO with KL loss 0.005 (low-variance KL), clipping 0.2 / 0.28, no entropy bonus;
Adam, constant learning rate 5e-7, weight decay 0.1; policy temperature 0.7, top-p 1.0; simulator temperature 0.7;
at most 10 utterances of at most 50 words, the persuader speaks first, a dialogue ends at the first donation marker
(eager termination) or after 5 exchanges; reward = donated amount / $2, capped at 1. SimMem: at most
N_aud = 4 audited groups per step, gate mean reward >= 0.9 and population std <= 0.1, the N_mem = 12 most recent
distinct rules in the simulator prompt, one rule with at most two examples per audited group.

A run directory `runs/p4g_4b_<arm>_<stamp>/` contains `ckpt/iter_XXXXXXX` (every 25 steps), `trajectories/rollout_<step>.jsonl`
(all 128 dialogues of each step) and, for SimMem, `trajectories/sim_memory/`:

* `memory.json`: every rule with its examples, cause, source group and step;
* `audit_log.jsonl`: every judge call (group, mean/std, raw output, parsed verdict);
* `events.jsonl`: one `snapshot` event per step (number of rules, size of the rendered block) and one `rule_added` event per rule.

`results/qwen3_4b_simmem/sim_memory/` holds these three files for the paper's run (used by the tests and by the
frozen-rules ablation below). `python -m usim.simmem.offline --run <run dir> --out <dir>
--judge llm --judge-model ... --judge-base-url ...` replays the memory loop over the saved trajectories of a run.

**Ablations** (paper, Table "Ablations of SimMem"). All variants branch from the GRPO run's checkpoint at iteration 99:

```bash
G=runs/p4g_4b_grpo_<stamp>/ckpt/iter_0000099
# w/o collapse check (evaluated at its last iteration, 149)
ARM=simmem FROM_CKPT=$G RESTART=finetune NUM_ROLLOUT=150 bash scripts/train_p4g_4b.sh --simmem-no-collapse-check
# frozen rules: the final rules of the full run from the start, never audited (evaluated at iteration 149)
ARM=simmem FROM_CKPT=$G RESTART=finetune NUM_ROLLOUT=150 bash scripts/train_p4g_4b.sh \
    --simmem-init results/qwen3_4b_simmem/sim_memory/memory.json --simmem-frozen
# raw contexts: the flagged dialogue verbatim instead of a rule, 8 most recent rendered (resumes; evaluated at 249)
ARM=simmem FROM_CKPT=$G RESTART=resume bash scripts/train_p4g_4b.sh --simmem-raw-context --simmem-raw-max-notes 8
```

## 4. Held-out evaluation (Table 1)

Each final checkpoint (iteration 249) is evaluated with one dialogue per test task (200 tasks) at policy temperature
0.7, eager termination, against persuadees never used for training or judging; the evaluation simulators receive no
memory.

```bash
export SIM_API_KEY=...
for spec in gpt4omini:openai/gpt-4o-mini doubaoseedmini:openai/Doubao-Seed-2.0-mini \
            ministral14b:openai/Ministral-3-14B-Instruct-2512; do
  RUN_DIR=runs/p4g_4b_simmem_<stamp> ITER=249 SIM_NAME=${spec%%:*} SIM_MODEL=${spec#*:} \
  SIM_BASE_URL=https://<provider>/v1 SIM_KEY_VAR=SIM_API_KEY bash scripts/eval_heldout.sh
done
# Zero-shot row: leave RUN_DIR unset (evaluates the untrained policy)
python3 scripts/score_eval.py runs/eval_*/trajectories/eval/*/rollout_000000.jsonl
```

Model ids differ between providers; use the id under which your provider serves each model.
`score_eval.py` prints the paper's metrics: **Deal** = share of dialogues with a donation and **Reward** = mean of
min(amount / $2, 1), where only an explicit `[DONATE $x]` / `[GIVE $x]` / `[$x]` marker in a persuadee turn counts
(the last one wins). It also prints the reward stored during the rollout, which additionally accepts
natural-language amounts. Both training and evaluation sample from LLM endpoints, so reruns vary somewhat.

## Notes

* SimMem is enabled only by `--simmem-enable`; without it `train.py` is plain multi-turn GRPO.
* The memory is shared by all personas and never enters evaluation (`usim/p4g/rollout.py` builds evaluation
  environments without it).
* The group gate uses the population std (ddof = 0); GRPO's advantage normalization uses the sample std (ddof = 1).
* Program-enforced constraints are: example fields non-empty, examples whose `better` contains a donation marker or a
  dollar amount are dropped, at most two examples, exact-text rule deduplication. Everything else the judge prompt asks
  for (rule scope, wording) is up to the judge.

## License

Apache-2.0 (see `LICENSE`). `data/p4g` is derived from the ConvoKit release of Persuasion for Good
(Wang et al., 2019); slime, Megatron-LM, SGLang and the models keep their own licenses.
