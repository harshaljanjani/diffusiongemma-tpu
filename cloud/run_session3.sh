#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/dgemma/venv}
PYTHON=$VENV/bin/python
RESULTS=${RESULTS:-$REPO_DIR/results/session3}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-2400}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

echo "=== download start"
"$VENV/bin/hf" download google/gemma-4-26B-A4B-it > "$RESULTS/download_gemma4.log" 2>&1 &
"$VENV/bin/hf" download google/diffusiongemma-26B-A4B-it > "$RESULTS/download_diffusion.log" 2>&1 &
wait
echo "=== weights ready seconds=$SECONDS"

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

suite autoregressive_headline --family gemma4 --batch-sizes 1 4 --timed-runs 2
suite diffusion_headline --family diffusion --batch-sizes 1 4 --timed-runs 2
suite autoregressive_ladder --family gemma4 --batch-sizes 16 2 8 --variants sampled --timed-runs 2
suite diffusion_ladder --family diffusion --batch-sizes 16 8 2 --timed-runs 2

echo "=== session 3 finished"
