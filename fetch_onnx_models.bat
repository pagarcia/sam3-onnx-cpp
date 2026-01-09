@echo off
REM sam3-onnx-cpp/fetch_onnx_models.bat
setlocal EnableExtensions

REM Download SAM3-Tracker ONNX models from Hugging Face (onnx-community)
REM Requires: python + huggingface_hub installed in your current environment.

set "REPO_ID=onnx-community/sam3-tracker-ONNX"
set "OUTDIR=checkpoints\sam3"

echo [INFO] Cleaning old "%OUTDIR%" ...
if exist "%OUTDIR%" rmdir /s /q "%OUTDIR%"

echo [INFO] Downloading ONNX files to "%OUTDIR%" ...
python -c "from huggingface_hub import hf_hub_download; import os; repo=r'%REPO_ID%'; out=r'%OUTDIR%'; files=['onnx/vision_encoder.onnx','onnx/vision_encoder.onnx_data','onnx/prompt_encoder_mask_decoder.onnx','onnx/prompt_encoder_mask_decoder.onnx_data']; os.makedirs(out, exist_ok=True); [hf_hub_download(repo_id=repo, filename=f, local_dir=out, local_dir_use_symlinks=False) for f in files]; print('[OK] Downloaded:\n  ' + '\n  '.join(files))"

if errorlevel 1 (
  echo [ERROR] Download failed. Make sure the venv is activated and huggingface_hub is installed.
  exit /b 1
)

echo [OK] Done.
pause
