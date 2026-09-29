#!/usr/bin/env bash
# Evaluate one saved checkpoint against one held-out persuadee (paper Table 1: 200 test tasks x 1 dialogue,
# eager termination, policy temperature 0.7, no simulator memory).
#
#   RUN_DIR=runs/p4g_4b_simmem_<stamp> ITER=249 \
#   SIM_NAME=gpt4omini SIM_MODEL=openai/gpt-4o-mini SIM_BASE_URL=https://api.openai.com/v1 SIM_KEY_VAR=OPENAI_API_KEY \
#     bash scripts/eval_heldout.sh
#
# The three held-out persuadees of the paper (any OpenAI-compatible provider; LiteLLM naming "openai/<model id>"):
#   gpt4omini       gpt-4o-mini
#   doubaoseedmini  Doubao-Seed-2.0-mini
#   ministral14b    Ministral-3-14B-Instruct-2512
# ITER=0 with RUN_DIR unset evaluates the untrained policy (Zero-shot row).
# Output: $OUT_DIR/trajectories/eval/p4g_<SIM_NAME>_eager/rollout_000000.jsonl; score it with scripts/score_eval.py.
# Needs 3 GPUs (2 actor GPUs only push the weights to SGLang, 1 SGLang GPU generates).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
SIM_NAME="${SIM_NAME:?e.g. gpt4omini}"
SIM_MODEL="${SIM_MODEL:?e.g. openai/gpt-4o-mini}"
SIM_BASE_URL="${SIM_BASE_URL:?OpenAI-compatible base URL of the held-out persuadee}"
SIM_KEY_VAR="${SIM_KEY_VAR:-OPENAI_API_KEY}"
: "${!SIM_KEY_VAR:?export $SIM_KEY_VAR=<API key of the held-out persuadee>}"
ITER="${ITER:-249}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3-4B-Instruct-2507}"
REF_DIR="${REF_DIR:-$ROOT/models/Qwen3-4B-Instruct-2507_torch_dist}"
SLIME_DIR="${SLIME_DIR:-$ROOT/third_party/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
export SLIME_DIR
OUT_DIR="${OUT_DIR:-$ROOT/runs/eval_${SIM_NAME}_$(basename "${RUN_DIR:-zeroshot}")_iter${ITER}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

if [ -n "${RUN_DIR:-}" ]; then
    # Load exactly one iteration: a private checkpoint root that holds a link to it.
    NAME="iter_$(printf '%07d' "$ITER")"
    [ -d "$RUN_DIR/ckpt/$NAME" ] || { echo "missing checkpoint $RUN_DIR/ckpt/$NAME" >&2; exit 2; }
    mkdir -p "$OUT_DIR/load"
    ln -sfn "$(cd "$RUN_DIR/ckpt/$NAME" && pwd -P)" "$OUT_DIR/load/$NAME"
    echo "$ITER" > "$OUT_DIR/load/latest_checkpointed_iteration.txt"
    LOAD_ARGS=(--load "$OUT_DIR/load" --no-load-optim --no-load-rng --finetune)
else
    LOAD_ARGS=(--load "$REF_DIR" --no-load-optim --no-load-rng --finetune)
fi

sed -e "s|__NAME__|$SIM_NAME|" -e "s|__MODEL__|$SIM_MODEL|" -e "s|__BASE_URL__|$SIM_BASE_URL|" \
    -e "s|__KEY_VAR__|$SIM_KEY_VAR|" configs/eval/heldout_eager.yaml.template > "$OUT_DIR/eval.yaml"

export CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-0,1,2}"
export PYTHONPATH="$ROOT:$SLIME_DIR:$MEGATRON_ROOT:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export USIM_MAX_CONCURRENCY="${USIM_MAX_CONCURRENCY:-16}"   # dialogues in flight against the API
export USIM_API_MAX_RETRIES="${USIM_API_MAX_RETRIES:-4}"
export USIM_API_TIMEOUT="${USIM_API_TIMEOUT:-90}"
export TENSORBOARD_DIR="$OUT_DIR/tensorboard"

source "$SLIME_DIR/scripts/models/qwen3-4B-Instruct-2507.sh"
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1
    --hf-checkpoint "$MODEL_DIR" --ref-load "$REF_DIR" "${LOAD_ARGS[@]}"
    --save "$OUT_DIR/ckpt" --save-interval 9999
    --num-rollout 0 --eval-interval 1 --eval-config "$OUT_DIR/eval.yaml"
    --rollout-batch-size 16 --n-samples-per-prompt 8 --global-batch-size 128
    --rollout-max-response-len 22528 --rollout-temperature 0.7 --balance-data
    --trainable-role agent --max-turns 10
    --usim-fixed-opponent-model "$SIM_MODEL" --usim-fixed-opponent-base-url "$SIM_BASE_URL"
    --usim-fixed-opponent-api-key-var "$SIM_KEY_VAR"
    --p4g-corpus-path "$ROOT/data/p4g/corpus" --p4g-dataset-dir "$ROOT/data/p4g/train"
    --p4g-eval-dataset-dir "$ROOT/data/p4g/test"
    --p4g-word-limit 50 --p4g-num-turns 10 --p4g-termination-mode eager
    --trajectory-output-dir "$OUT_DIR/trajectories"
    --tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1
    --context-parallel-size 2 --micro-batch-size 8
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
    --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --sglang-mem-fraction-static 0.7
    --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash --use-tensorboard
)
python3 train.py "${MODEL_ARGS[@]}" "${ARGS[@]}" "$@" 2>&1 | tee "$OUT_DIR/console.log"
python3 scripts/score_eval.py "$OUT_DIR"/trajectories/eval/*/rollout_000000.jsonl
