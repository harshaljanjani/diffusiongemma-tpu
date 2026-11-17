#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/dgemma/venv}
PYTHON=$VENV/bin/python
RESULTS=${RESULTS:-$REPO_DIR/results/session2}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-3600}
BATCH_SIZES=${BATCH_SIZES:-1 4 16 2 8 32}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

echo "=== download start"
"$PYTHON" -m pytest -q "$REPO_DIR/tests" > "$RESULTS/unit_tests.log" 2>&1 &
"$VENV/bin/hf" download google/diffusiongemma-26B-A4B-it > "$RESULTS/download_diffusion.log" 2>&1 &
"$VENV/bin/hf" download google/gemma-4-26B-A4B-it > "$RESULTS/download_gemma4.log" 2>&1 &
"$PYTHON" -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='test')" > "$RESULTS/download_gsm8k.log" 2>&1
wait
echo "=== download done seconds=$SECONDS"
du -sh /opt/hf-cache/hub/models--* 2> /dev/null

eval "$("$PYTHON" -m torch.tpu.distributed | sed 's/^/export /')"
echo "WORLD_SIZE=${WORLD_SIZE:-unset} TORCH_TPU_TOPOLOGY=${TORCH_TPU_TOPOLOGY:-unset}"

suite() {
  local name=$1
  shift
  echo "=== stage $name"
  local start=$SECONDS
  timeout "$STAGE_TIMEOUT" "$VENV/bin/torchrun" --nproc_per_node="$WORLD_SIZE" run_suite.py \
    --expert-parallel --output "$RESULTS/results.jsonl" "$@" > "$RESULTS/$name.log" 2>&1
  echo "=== stage $name exit=$? seconds=$((SECONDS - start))"
}

suite diffusion_smoke --family diffusion --batch-sizes 1 --timed-runs 1 --label smoke_
suite autoregressive_smoke --family gemma4 --batch-sizes 1 --timed-runs 1 --label smoke_
suite diffusion --family diffusion --batch-sizes $BATCH_SIZES --variants grouped check1 check0 --gsm8k-examples 50
suite autoregressive --family gemma4 --batch-sizes $BATCH_SIZES --variants grouped sampled --gsm8k-examples 50

echo "=== session 2 finished"
