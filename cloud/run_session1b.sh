#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-/opt/dgemma/venv/bin/python}
RESULTS=${RESULTS:-$REPO_DIR/results/session1b}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-1200}
PROXY_MODEL=${PROXY_MODEL:-proxy:6}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

run() {
  local script=$1 name=$2
  shift 2
  echo "=== stage $name"
  local start=$SECONDS
  timeout "$STAGE_TIMEOUT" "$PYTHON" "$script" --output "$RESULTS/results.jsonl" --label "$name" "$@" > "$RESULTS/$name.log" 2>&1
  echo "=== stage $name exit=$? seconds=$((SECONDS - start))"
}

DIFFUSION="--model $PROXY_MODEL --max-new-tokens 512 --prompt-length 200 --sampler static --compile auto --timed-runs 2"
AUTOREGRESSIVE="--model $PROXY_MODEL --max-new-tokens 256 --prompt-length 200 --sampler static --compile auto --timed-runs 2"

run run_diffusion.py diffusion_grouped_b1 $DIFFUSION --experts-implementation grouped_mm
for BATCH in 1 2 4 8 16; do
  run run_diffusion.py "diffusion_by_token_count_b$BATCH" $DIFFUSION --experts-implementation by_token_count --batch-size "$BATCH"
done
run run_ar.py autoregressive_grouped_b1 $AUTOREGRESSIVE --experts-implementation grouped_mm
for BATCH in 1 2 4 8 16; do
  run run_ar.py "autoregressive_by_token_count_b$BATCH" $AUTOREGRESSIVE --experts-implementation by_token_count --batch-size "$BATCH"
done
run run_ar.py autoregressive_batched_b4 $AUTOREGRESSIVE --experts-implementation batched_mm --batch-size 4 --prompt-length 8 --prompt-bucket 8

echo "=== session 1b finished"
