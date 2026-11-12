#!/usr/bin/env bash
set -uxo pipefail
exec > /var/log/dgemma-bootstrap.log 2>&1

WORK_DIR=/opt/dgemma
mkdir -p "$WORK_DIR" /opt/hf-cache /opt/tpu-compile-cache
chmod -R 777 "$WORK_DIR" /opt/hf-cache /opt/tpu-compile-cache

export HOME=/root
export UV_INSTALL_DIR=/usr/local/bin
export UV_PYTHON_INSTALL_DIR=/opt/dgemma/python
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 "$WORK_DIR/venv"
uv pip install --python "$WORK_DIR/venv/bin/python" \
  "torch_tpu==0.2.0" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$WORK_DIR/venv/bin/python" \
  "transformers==5.22.2" "diffusers==0.42.1" accelerate datasets hf_transfer pytest \
  xprof tensorboard tensorboard-plugin-profile "setuptools<81"

chmod -R 777 "$WORK_DIR"
"$WORK_DIR/venv/bin/python" -c "import torch; print(torch.__version__, torch.accelerator.is_available(), torch.accelerator.device_count())"
touch "$WORK_DIR/BOOTSTRAP_DONE"
