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
