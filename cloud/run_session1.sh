#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-/opt/dgemma/venv/bin/python}
RESULTS=${RESULTS:-$REPO_DIR/results/session1}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-1200}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/opt/tpu-compile-cache
cd "$REPO_DIR/bench"

stage() {
  local name=$1
  shift
  echo "=== stage $name: $*"
  local start=$SECONDS
  timeout "$STAGE_TIMEOUT" "$@" > "$RESULTS/$name.log" 2>&1
  echo "=== stage $name exit=$? seconds=$((SECONDS - start))"
  tail -n 3 "$RESULTS/$name.log" | cut -c1-600
}

diffusion() {
  local name=$1
  shift
  stage "$name" "$PYTHON" run_diffusion.py --output "$RESULTS/diffusion.jsonl" --label "$name" "$@"
}

autoregressive() {
  local name=$1
  shift
  stage "$name" "$PYTHON" run_ar.py --output "$RESULTS/autoregressive.jsonl" --label "$name" "$@"
}

stage environment "$PYTHON" -c "
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__)
print('accelerator', torch.accelerator.current_accelerator(), torch.accelerator.device_count())
import torch_tpu
print('torch_tpu', getattr(torch_tpu, '__version__', 'unknown'))
x = torch.ones(4, device='tpu')
print((x * 2).sum().item())
"
stage unit_tests "$PYTHON" -m pytest -q "$REPO_DIR/tests"
stage op_coverage "$PYTHON" op_coverage.py --output "$RESULTS/op_coverage.jsonl"

TINY="--model tiny --max-new-tokens 512 --prompt-length 40"
diffusion tiny_static_compiled $TINY --sampler static --compile auto
diffusion tiny_static_eager $TINY --sampler static --compile none --timed-runs 1 --warmup-runs 1
diffusion tiny_transformers_static $TINY --sampler transformers --cache static --timed-runs 1 --warmup-runs 2
diffusion tiny_transformers_dynamic $TINY --sampler transformers --cache dynamic --compile none --timed-runs 1 --warmup-runs 1

PROXY="--model ${PROXY_MODEL:-proxy:6} --max-new-tokens 512 --prompt-length 200"
diffusion proxy_static_compiled_b1 $PROXY --sampler static --compile auto --batch-size 1
diffusion proxy_static_check1 $PROXY --sampler static --compile auto --stop-check-interval 1
diffusion proxy_static_check0 $PROXY --sampler static --compile auto --stop-check-interval 0
diffusion proxy_static_eager $PROXY --sampler static --compile none --timed-runs 1 --warmup-runs 1
TORCH_TPU_DEFER_AND_FUSE=1 diffusion proxy_static_defer_and_fuse $PROXY --sampler static --compile none --timed-runs 1 --warmup-runs 1
diffusion proxy_transformers_static $PROXY --sampler transformers --cache static --timed-runs 1 --warmup-runs 2
diffusion proxy_transformers_dynamic $PROXY --sampler transformers --cache dynamic --compile none --timed-runs 1 --warmup-runs 1
for BATCH in 2 4 8; do
  diffusion "proxy_static_compiled_b$BATCH" $PROXY --sampler static --compile auto --batch-size "$BATCH" --timed-runs 2
done
for EXPERTS in eager batched_mm grouped_mm; do
  diffusion "proxy_static_experts_$EXPERTS" $PROXY --sampler static --compile auto --experts-implementation "$EXPERTS" --timed-runs 2
done
for ATTENTION in eager sdpa; do
  diffusion "proxy_static_attention_$ATTENTION" $PROXY --sampler static --compile auto --attn-implementation "$ATTENTION" --timed-runs 2
done

AR_PROXY="--model ${PROXY_MODEL:-proxy:6} --max-new-tokens 256 --prompt-length 200"
autoregressive ar_proxy_static_compiled_b1 $AR_PROXY --sampler static --compile auto
autoregressive ar_proxy_static_eager $AR_PROXY --sampler static --compile none --max-new-tokens 64 --timed-runs 1 --warmup-runs 1
autoregressive ar_proxy_transformers $AR_PROXY --sampler transformers --timed-runs 1 --warmup-runs 2
for BATCH in 4 8; do
  autoregressive "ar_proxy_static_compiled_b$BATCH" $AR_PROXY --sampler static --compile auto --batch-size "$BATCH" --timed-runs 2
done

stage xprof "$PYTHON" xprof_capture.py --model "${PROXY_MODEL:-proxy:6}" --output-dir "$RESULTS/xprof"

echo "=== session 1 finished"
