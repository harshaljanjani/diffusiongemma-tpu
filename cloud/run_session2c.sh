#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/dgemma/venv}
PYTHON=$VENV/bin/python
RESULTS=${RESULTS:-$REPO_DIR/results/session2c}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-2400}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

eval "$("$PYTHON" -m torch.tpu.distributed | sed 's/^/export /')"

suite() {
  local name=$1
  shift
  echo "=== stage $name"
  local start=$SECONDS
  timeout "$STAGE_TIMEOUT" "$VENV/bin/torchrun" --nproc_per_node="$WORLD_SIZE" run_suite.py \
    --expert-parallel --output "$RESULTS/results.jsonl" "$@" > "$RESULTS/$name.log" 2>&1
  echo "=== stage $name exit=$? seconds=$((SECONDS - start))"
}

suite autoregressive --family gemma4 --batch-sizes 1 4 16 2 8 --timed-runs 2
suite diffusion --family diffusion --batch-sizes 1 16 8 4 --timed-runs 2

echo "=== session 2c finished"
