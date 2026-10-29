#!/usr/bin/env bash
set -euo pipefail
NEXT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Pass an exported checkpoint directory}"
export HF_HOME="${HF_HOME:-/tmp/shopping-next-cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/tmp/shopping-next-cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/shopping-next-cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/shopping-next-cache/xdg}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${SHOPPING_NEXT_RUNTIME:-/tmp/shopping-next-cache}/flashinfer}"
export MAX_JOBS="${MAX_JOBS:-8}"
CONTEXT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["budgets"]["context_tokens"])' "$NEXT_ROOT/configs/execution.json")"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "${SERVED_MODEL_NAME:-shopping-agent}" \
  --host 127.0.0.1 --port "${LLM_PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --gdn-prefill-backend triton --max-num-seqs 32 \
  --max-model-len "$CONTEXT" --enable-auto-tool-choice --tool-call-parser qwen3_coder
