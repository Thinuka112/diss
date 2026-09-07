#!/usr/bin/env bash
# Set up an isolated environment on a Hex node and run the slice-quality training.
# Run this from ~/quality on the node (after the files have been transferred up).
#
#   cd ~/quality
#   bash setup_and_run.sh
#
# Hex nodes are shared, so everything goes in a local venv - nothing touches the
# system Python. Remember: the cloud has NO backups, so copy models/ back to your
# laptop when the run finishes.
set -euo pipefail

# 1. Isolated Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. PyTorch matched to the node's CUDA. Check `nvidia-smi` (top-right shows the
#    CUDA version) and set CUDA_TAG to cu121 / cu118 / etc. cu121 is a safe default.
CUDA_TAG="${CUDA_TAG:-cu121}"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# 3. The remaining deps
pip install -r requirements.txt

# 4. Confirm the GPU is actually visible before spending time on a run
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("CUDA available:", ok)
if ok:
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    raise SystemExit("No GPU visible - check you are on a GPU node (nvidia-smi).")
PY

# 5. Train at higher resolution via env overrides (no code edit needed).
#    Defaults below suit an RTX 3080 (10 GB, the 'Everyone' nodes garlick/ogg).
#    On an A100 80 GB (request 'carrot' via the Hex list) push to 768 or 1024.
#    If you hit CUDA out-of-memory, lower QUALITY_BATCH first, then QUALITY_INPUT_SIZE.
export QUALITY_INPUT_SIZE="${QUALITY_INPUT_SIZE:-640}"
export QUALITY_BATCH="${QUALITY_BATCH:-8}"
echo "Training at ${QUALITY_INPUT_SIZE}px, batch ${QUALITY_BATCH}"
python train_quality_cnn.py
