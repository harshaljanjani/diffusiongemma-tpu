#!/usr/bin/env bash
set -uo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-/opt/dgemma/venv}
PYTHON=$VENV/bin/python
RESULTS=${RESULTS:-$REPO_DIR/results/session4}
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_DIR"
export HF_HOME=/opt/hf-cache
cd "$REPO_DIR/bench"

echo "=== stage xprof_autoregressive"
for BATCH in 1 16; do
  timeout 900 "$PYTHON" xprof_capture.py --family gemma4 --model proxy:6 --batch-size "$BATCH" --max-new-tokens 24 \
    --experts-implementation by_token_count --output-dir "$RESULTS/xprof_autoregressive_b$BATCH" \
    > "$RESULTS/xprof_autoregressive_b$BATCH.log" 2>&1
  echo "=== xprof batch $BATCH exit=$?"
  gzip -f "$RESULTS/xprof_autoregressive_b$BATCH"/*.json
done

echo "=== stage notebook"
"$VENV/bin/python" -m pip --version > /dev/null 2>&1 || /usr/local/bin/uv pip install --python "$PYTHON" pip
/usr/local/bin/uv pip install --python "$PYTHON" nbconvert ipykernel pandas matplotlib jinja2 > "$RESULTS/notebook_install.log" 2>&1
/usr/local/bin/uv pip install --python "$PYTHON" "$REPO_DIR" >> "$RESULTS/notebook_install.log" 2>&1
NOTEBOOK=rebuilding_diffusiongemma_sampler_tpu
sed '/"!pip install/d' "$REPO_DIR/notebooks/$NOTEBOOK.ipynb" > "$RESULTS/$NOTEBOOK.input.ipynb"
cd "$RESULTS"
timeout 3000 "$PYTHON" -m jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  --output "$NOTEBOOK.executed.ipynb" "$NOTEBOOK.input.ipynb" > "$RESULTS/notebook_run.log" 2>&1
echo "=== notebook exit=$?"

echo "=== session 4 finished"
