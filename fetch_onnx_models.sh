#!/usr/bin/env bash
# sam3-onnx-cpp/fetch_onnx_models.sh
set -euo pipefail

REPO_ID="onnx-community/sam3-tracker-ONNX"
OUTDIR="checkpoints/sam3"

VARIANT="${1:-fp32}"   # fp32 | fp16 | clean

if [[ "${VARIANT}" == "clean" ]]; then
  echo "[INFO] Cleaning ${OUTDIR} ..."
  rm -rf "${OUTDIR}"
  echo "[OK] Cleaned."
  exit 0
fi

mkdir -p "${OUTDIR}"

ENC="onnx/vision_encoder.onnx"
ENC_DATA="onnx/vision_encoder.onnx_data"
DEC="onnx/prompt_encoder_mask_decoder.onnx"
DEC_DATA="onnx/prompt_encoder_mask_decoder.onnx_data"

if [[ "${VARIANT}" == "fp16" ]]; then
  ENC="onnx/vision_encoder_fp16.onnx"
  ENC_DATA="onnx/vision_encoder_fp16.onnx_data"
  DEC="onnx/prompt_encoder_mask_decoder_fp16.onnx"
  DEC_DATA="onnx/prompt_encoder_mask_decoder_fp16.onnx_data"
elif [[ "${VARIANT}" != "fp32" ]]; then
  echo "[WARN] Unknown variant '${VARIANT}'. Using fp32."
  VARIANT="fp32"
fi

echo "[INFO] Variant: ${VARIANT}"
echo "[INFO] Downloading to ${OUTDIR} ..."

python -c "
from huggingface_hub import hf_hub_download
import os
repo='${REPO_ID}'
out='${OUTDIR}'
files=['${ENC}','${ENC_DATA}','${DEC}','${DEC_DATA}']
os.makedirs(out, exist_ok=True)
for f in files:
  hf_hub_download(repo_id=repo, filename=f, local_dir=out)
print('[OK] Downloaded:\\n  ' + '\\n  '.join(files))
"

echo "[OK] Done."
