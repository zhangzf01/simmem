#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 persuader on P4G against a frozen Qwen2.5-72B-Instruct-AWQ simulator.
#
#   ARM=simmem bash scripts/train_p4g_4b.sh     # GRPO + SimMem (the paper's method)
#   ARM=grpo   bash scripts/train_p4g_4b.sh     # GRPO baseline: identical except --simmem-enable
#
# Needs (see README): the simulator served at $SIM_BASE_URL (scripts/serve_simulator.sh), the SimMem judge at
# $JUDGE_BASE_URL (simmem arm only), and the torch_dist copy of the policy (scripts/convert_hf_to_torch_dist.sh).
# Layout of the paper runs on one 8xH20 node: simulator on GPUs 0-3 (tp=4); training on GPUs 4-7
# (2 actor GPUs with context parallel 2 + 2 SGLang rollout GPUs). 250 steps took ~14 h.
#
# Short smoke test first (~10 min):  ARM=simmem NUM_ROLLOUT=2 SAVE_INTERVAL=1 bash scripts/train_p4g_4b.sh
# Extra flags are passed through to train.py, e.g. the paper's ablations:
#   --simmem-no-collapse-check     (no collapse check: every gated group is audited as collapsed)
#   --simmem-raw-context           (store the flagged conversation instead of a rule + examples)
#   --simmem-init <memory.json> --simmem-frozen   (fixed memory from the start, no updates)
# Start from a saved checkpoint (the ablations branch from the GRPO run's iteration 99):
#   FROM_CKPT=<run>/ckpt/iter_0000099 RESTART=finetune   weights only; step ids restart at 0 (set NUM_ROLLOUT=150)
#   FROM_CKPT=<run>/ckpt/iter_0000099 RESTART=resume     weights + optimizer; step ids continue at 100
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

ARM="${ARM:?set ARM=simmem or ARM=grpo}"
case "$ARM" in simmem) ARM_ARGS=(--simmem-enable);; grpo) ARM_ARGS=();; *) echo "ARM must be simmem or grpo" >&2; exit 2;; esac

MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3-4B-Instruct-2507}"
REF_DIR="${REF_DIR:-$ROOT/models/Qwen3-4B-Instruct-2507_torch_dist}"
SLIME_DIR="${SLIME_DIR:-$ROOT/third_party/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
export SLIME_DIR

# Training simulator (persuadee) and SimMem judge: any OpenAI-compatible endpoints.
SIM_BASE_URL="${SIM_BASE_URL:-http://127.0.0.1:30010/v1}"
SIM_MODEL="${SIM_MODEL:-openai/qwen2.5-72b}"            # LiteLLM naming: openai/<served model name>
# Judge (simmem arm): the paper uses DeepSeek-V4-Flash with greedy decoding. JUDGE_MODEL is the model id the
# endpoint expects; JUDGE_API_KEY_VAR names the env var that holds its key.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY_VAR="${JUDGE_API_KEY_VAR:-JUDGE_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"         # key of the (local) simulator endpoint

NUM_ROLLOUT="${NUM_ROLLOUT:-250}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"      # periodic eval during training; needs EVAL_CONFIG (see configs/eval)
ROLLOUT_BS=16; M=8                       # 16 tasks x 8 rollouts per step
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/p4g_4b_${ARM}_$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-4,5,6,7}"
export PYTHONPATH="$ROOT:$SLIME_DIR:$MEGATRON_ROOT:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export USIM_API_MAX_RETRIES="${USIM_API_MAX_RETRIES:-2}"   # retries of a failed simulator call
export USIM_API_TIMEOUT="${USIM_API_TIMEOUT:-60}"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
export no_proxy="127.0.0.1,localhost,0.0.0.0,${no_proxy:-}"
export NO_PROXY="$no_proxy"

[ -d "$OUTPUT_DIR/ckpt" ] && { echo "OUTPUT_DIR already has checkpoints: $OUTPUT_DIR" >&2; exit 2; }
[ -d "$REF_DIR" ] || { echo "missing $REF_DIR; run scripts/convert_hf_to_torch_dist.sh first" >&2; exit 2; }
curl -sf -m 10 "$SIM_BASE_URL/models" >/dev/null || { echo "simulator not reachable at $SIM_BASE_URL" >&2; exit 3; }

source "$SLIME_DIR/scripts/models/qwen3-4B-Instruct-2507.sh"   # defines MODEL_ARGS
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1
    --hf-checkpoint "$MODEL_DIR" --ref-load "$REF_DIR" --save "$OUTPUT_DIR/ckpt"
    --save-interval "$SAVE_INTERVAL" --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BS" --n-samples-per-prompt "$M" --global-batch-size $((ROLLOUT_BS * M))
    --rollout-max-response-len 16384 --rollout-temperature 0.7 --balance-data
    --trainable-role agent
    --usim-fixed-opponent-model "$SIM_MODEL" --usim-fixed-opponent-base-url "$SIM_BASE_URL"
    --usim-fixed-opponent-api-key-var OPENAI_API_KEY
    --p4g-corpus-path "$ROOT/data/p4g/corpus" --p4g-dataset-dir "$ROOT/data/p4g/train"
    --p4g-eval-dataset-dir "$ROOT/data/p4g/test"
    --p4g-word-limit 50 --p4g-num-turns 10 --p4g-termination-mode eager
    --trajectory-output-dir "$OUTPUT_DIR/trajectories"
    --rollout-concurrency 64
    --tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1
    --context-parallel-size 2 --micro-batch-size 8
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.005 --kl-loss-type low_var_kl
    --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
    --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --sglang-mem-fraction-static 0.7
    --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash --use-tensorboard
)
if [ "$ARM" = simmem ]; then
    [ -n "$JUDGE_BASE_URL" ] && [ -n "$JUDGE_MODEL" ] || { echo "set JUDGE_BASE_URL and JUDGE_MODEL" >&2; exit 2; }
    # N_aud=4 groups/step with mean >= 0.9 and population std <= 0.1; N_mem=12 rules in the prompt.
    ARGS+=(--simmem-judge-model "$JUDGE_MODEL" --simmem-judge-base-url "$JUDGE_BASE_URL"
           --simmem-judge-api-key-var "$JUDGE_API_KEY_VAR"
           --simmem-mean-threshold 0.9 --simmem-std-threshold 0.1
           --simmem-max-audits-per-step 4 --simmem-max-notes 12)
    : "${!JUDGE_API_KEY_VAR:?export $JUDGE_API_KEY_VAR=<judge API key> (use any value for a local endpoint)}"
    curl -sf -m 10 -H "Authorization: Bearer ${!JUDGE_API_KEY_VAR}" "$JUDGE_BASE_URL/models" >/dev/null ||
        echo "warning: GET $JUDGE_BASE_URL/models failed; the judge must answer /chat/completions" >&2
fi
if [ -n "${FROM_CKPT:-}" ]; then
    # A private checkpoint root that holds a link to the one iteration to start from.
    FROM_ITER=$((10#$(basename "$FROM_CKPT" | sed 's/^iter_//')))
    mkdir -p "$OUTPUT_DIR/load"
    ln -sfn "$(cd "$FROM_CKPT" && pwd -P)" "$OUTPUT_DIR/load/$(basename "$FROM_CKPT")"
    echo "$FROM_ITER" > "$OUTPUT_DIR/load/latest_checkpointed_iteration.txt"
    ARGS+=(--load "$OUTPUT_DIR/load")
    case "${RESTART:-resume}" in
        resume) ;;
        finetune) ARGS+=(--no-load-optim --no-load-rng --finetune);;
        *) echo "RESTART must be resume or finetune" >&2; exit 2;;
    esac
fi
if [ "$EVAL_INTERVAL" != 0 ]; then
    ARGS+=(--eval-interval "$EVAL_INTERVAL" --eval-config "${EVAL_CONFIG:?set EVAL_CONFIG to an eval yaml}")
fi

mkdir -p "$OUTPUT_DIR"
echo "ARM=$ARM  output=$OUTPUT_DIR  simulator=$SIM_MODEL@$SIM_BASE_URL"
python3 train.py "${MODEL_ARGS[@]}" "${ARGS[@]}" ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} "$@" 2>&1 | tee "$OUTPUT_DIR/console.log"
