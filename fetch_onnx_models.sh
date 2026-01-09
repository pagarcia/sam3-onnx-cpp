#!/usr/bin/env bash
# sam3-onnx-cpp/fetch_onnx_models.sh

set -euo pipefail

# Download SAM3-Tracker ONNX models from Hugging Face (onnx-community)
# Requires: python + huggingface_hub installed in your current environment.

REPO_ID="onnx-community/sam3-tracker-ONNX"
OUTDIR="checkpoints/sam3"

echo "[INFO] Cleaning old ${OUTDIR} ..."
rm -rf "${OUTDIR}"

echo "[INFO] Downloading ONNX files to ${OUTDIR} ..."
python -c "
from huggingface_hub import hf_hub_download
repo='${REPO_ID}'
out='${OUTDIR}'
files=[
  'onnx/vision_encoder.onnx',
  'onnx/vision_encoder.onnx_data',
  'onnx/prompt_encoder_mask_decoder.onnx',
  'onnx/prompt_encoder_mask_decoder.onnx_data',
]
for f in files:
  hf_hub_download(repo_id=repo, filename=f, local_dir=out, local_dir_use_symlinks=False)
print('[OK] Downloaded:\\n  ' + '\\n  '.join(files))
"

echo "[OK] Done."
