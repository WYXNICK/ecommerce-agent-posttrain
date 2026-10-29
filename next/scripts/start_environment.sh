#!/usr/bin/env bash
set -euo pipefail
NEXT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOP_ROOT="$NEXT_ROOT/../environments/ShopSimulator/shop_env"
SHOP_PYTHON="${SHOP_PYTHON:-$SHOP_ROOT/../.venv-shopsim/bin/python}"
export SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOP_ENV_CONFIG="$SHOP_ROOT/configs/environment.json"
export SHOP_SEARCH_INDEX="$SHOP_ROOT/search_engine/products.sqlite3"
export SHOP_MAX_STEPS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["budgets"]["max_steps"])' "$NEXT_ROOT/configs/execution.json")"
export SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-8}"
export SHOPSIM_PORT="${SHOPSIM_PORT:-5701}"
export PYTHONPATH="$SHOP_ROOT:$SHOP_ROOT/shop_env${PYTHONPATH:+:$PYTHONPATH}"
# ShopSimulator writes a relative log; isolate it from the reference source tree.
mkdir -p "${SHOP_RUNTIME_DIR:-/tmp/shopping-next-environment}"
cd "${SHOP_RUNTIME_DIR:-/tmp/shopping-next-environment}"
exec "$SHOP_PYTHON" -c 'import runpy,sys; runpy.run_path(sys.argv[1], run_name="__main__")' "$SHOP_ROOT/shop_env/pack_api.py"
