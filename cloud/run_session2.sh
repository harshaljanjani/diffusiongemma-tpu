#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/dgemma/venv}
PYTHON=$VENV/bin/python
RESULTS=${RESULTS:-$REPO_DIR/results/session2}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-1800}
DIFFUSION_ID=google/diffusiongemma-26B-A4B-it
GEMMA4_ID=google/gemma-4-26B-A4B-it
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

echo "=== download start"
"$VENV/bin/hf" download "$DIFFUSION_ID" > "$RESULTS/download_diffusion.log" 2>&1 &
"$VENV/bin/hf" download "$GEMMA4_ID" > "$RESULTS/download_gemma4.log" 2>&1 &
wait
echo "=== download done seconds=$SECONDS"
du -sh /opt/hf-cache/hub/* 2> /dev/null

eval "$("$PYTHON" -m torch.tpu.distributed | sed 's/^/export /')"
echo "WORLD_SIZE=${WORLD_SIZE:-unset} TORCH_TPU_TOPOLOGY=${TORCH_TPU_TOPOLOGY:-unset}"
export HF_HUB_OFFLINE=1

run() {
  local name=$1
  shift
  echo "=== stage $name"
  local start=$SECONDS
  timeout "$STAGE_TIMEOUT" "$VENV/bin/torchrun" --nproc_per_node="$WORLD_SIZE" "$@" --output "$RESULTS/results.jsonl" --label "$name" > "$RESULTS/$name.log" 2>&1
  echo "=== stage $name exit=$? seconds=$((SECONDS - start))"
}

DIFFUSION="run_diffusion.py --model $DIFFUSION_ID --expert-parallel --prompts chat --max-new-tokens 512 --prompt-bucket 64 --sampler static --compile auto"
AUTOREGRESSIVE="run_ar.py --model $GEMMA4_ID --expert-parallel --prompts chat --max-new-tokens 512 --prompt-bucket 64 --sampler static --compile auto"
BEST="--experts-implementation by_token_count"

run diffusion_b1 $DIFFUSION $BEST
run autoregressive_b1 $AUTOREGRESSIVE $BEST
run gsm8k_diffusion quality_gsm8k.py --family diffusion --expert-parallel --num-examples 50 --batch-size 10
run gsm8k_autoregressive quality_gsm8k.py --family gemma4 --expert-parallel --num-examples 50 --batch-size 10
for BATCH in 4 16; do
  run "diffusion_b$BATCH" $DIFFUSION $BEST --batch-size "$BATCH" --timed-runs 2
  run "autoregressive_b$BATCH" $AUTOREGRESSIVE $BEST --batch-size "$BATCH" --timed-runs 2
done
run diffusion_grouped_b1 $DIFFUSION --experts-implementation grouped_mm --timed-runs 2
run autoregressive_grouped_b1 $AUTOREGRESSIVE --experts-implementation grouped_mm --timed-runs 2
for BATCH in 2 8 32; do
  run "diffusion_b$BATCH" $DIFFUSION $BEST --batch-size "$BATCH" --timed-runs 2
  run "autoregressive_b$BATCH" $AUTOREGRESSIVE $BEST --batch-size "$BATCH" --timed-runs 2
done
run diffusion_check1_b1 $DIFFUSION $BEST --stop-check-interval 1 --timed-runs 2
run diffusion_check0_b1 $DIFFUSION $BEST --stop-check-interval 0 --timed-runs 2

echo "=== session 2 finished"
