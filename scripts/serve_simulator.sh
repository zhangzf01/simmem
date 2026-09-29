#!/usr/bin/env bash
# Frozen training simulator: Qwen2.5-72B-Instruct-AWQ behind SGLang's OpenAI-compatible server.
#   bash scripts/serve_simulator.sh                 # GPUs 0-3, tp=4, port 30010, served name qwen2.5-72b
#   SIM_GPUS=0,1 SIM_TP=2 bash scripts/serve_simulator.sh   # also works; slower
# The trainer calls it as model "openai/qwen2.5-72b" at http://127.0.0.1:30010/v1 (see train_p4g_4b.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SIM_MODEL_PATH="${SIM_MODEL_PATH:-$ROOT/models/Qwen2.5-72B-Instruct-AWQ}"
export CUDA_VISIBLE_DEVICES="${SIM_GPUS:-0,1,2,3}"
exec python3 -m sglang.launch_server \
  --model-path "$SIM_MODEL_PATH" --served-model-name qwen2.5-72b \
  --tp-size "${SIM_TP:-4}" --host 0.0.0.0 --port "${SIM_PORT:-30010}" \
  --mem-fraction-static 0.85 --context-length 32768 \
  --attention-backend fa3
