#!/usr/bin/env bash
set -euo pipefail

# Example: start a persistent vLLM OpenAI-compatible server
# Usage:
#   bash example_scripts/start_vllm_server.sh meta-llama/Llama-3.2-3B-Instruct 8000

MODEL_ID=${1:-meta-llama/Llama-3.2-3B-Instruct}
PORT=${2:-8000}
TP_SIZE=${TP_SIZE:-1}
GPU_UTIL=${GPU_UTIL:-0.8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
DP_SIZE=${DP_SIZE:-1}

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --data-parallel-size "$DP_SIZE" \
  --served-model-name "$MODEL_ID" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --quantization "bitsandbytes" \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --max_model_len 8192

