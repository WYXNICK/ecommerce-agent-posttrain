#!/usr/bin/env bash
set -euo pipefail
NEXT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${SHOP_RUNTIME:?Set the existing frozen runtime directory}/activate.sh"
export PYTHONPATH="$NEXT_ROOT/src:$NEXT_ROOT/scripts/sft:$NEXT_ROOT/scripts/grpo"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
ulimit -n 65536
exec python "$NEXT_ROOT/scripts/grpo/run.py" "$@"
