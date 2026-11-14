#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/dgemma/venv}
PYTHON=$VENV/bin/python
RESULTS=${RESULTS:-$REPO_DIR/results/session1c}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-1500}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

run() {
  local name=$1
  shift
  echo "=== stage $name"
  local start=$SECONDS
  timeout "$STAGE_TIMEOUT" "$@" --output "$RESULTS/results.jsonl" --label "$name" > "$RESULTS/$name.log" 2>&1
  echo "=== stage $name exit=$? seconds=$((SECONDS - start))"
}

COMMON="--sampler static --compile auto --timed-runs 2 --prompt-length 200 --experts-implementation by_token_count"

run diffusion_proxy6_tied "$PYTHON" run_diffusion.py --model proxy:6 --max-new-tokens 512 $COMMON
run autoregressive_proxy6_tied "$PYTHON" run_ar.py --model proxy:6 --max-new-tokens 256 $COMMON

eval "$("$PYTHON" -m torch.tpu.distributed | sed 's/^/export /')"
echo "WORLD_SIZE=${WORLD_SIZE:-unset} TORCH_TPU_TOPOLOGY=${TORCH_TPU_TOPOLOGY:-unset}"
run diffusion_proxy6_one_rank "$VENV/bin/torchrun" --nproc_per_node="${WORLD_SIZE:-1}" run_diffusion.py --model proxy:6 --max-new-tokens 512 $COMMON --expert-parallel
run autoregressive_proxy6_one_rank "$VENV/bin/torchrun" --nproc_per_node="${WORLD_SIZE:-1}" run_ar.py --model proxy:6 --max-new-tokens 256 $COMMON --expert-parallel

run diffusion_proxy12 "$PYTHON" run_diffusion.py --model proxy:12 --max-new-tokens 512 $COMMON
run autoregressive_proxy12 "$PYTHON" run_ar.py --model proxy:12 --max-new-tokens 256 $COMMON

echo "=== session 1c finished"
