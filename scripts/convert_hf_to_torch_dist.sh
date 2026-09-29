#!/usr/bin/env bash
# Megatron torch_dist copy of the policy (used as --ref-load; the actor starts from the same weights).
#   bash scripts/convert_hf_to_torch_dist.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SLIME_DIR="${SLIME_DIR:-$ROOT/third_party/slime}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3-4B-Instruct-2507}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
source "$SLIME_DIR/scripts/models/qwen3-4B-Instruct-2507.sh"
PYTHONPATH="$SLIME_DIR:${MEGATRON_ROOT:-/root/Megatron-LM}:${PYTHONPATH:-}" python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL_DIR" --save "${MODEL_DIR}_torch_dist"
