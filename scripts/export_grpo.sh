#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir]}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/export_grpo.py" \
  "$ACTOR_CHECKPOINT" "$OUTPUT_DIR"
