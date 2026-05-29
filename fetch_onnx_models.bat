@echo off
REM sam3-onnx-cpp/fetch_onnx_models.bat
setlocal EnableExtensions

set "REPO_ID=onnx-community/sam3-tracker-ONNX"
set "HF_REVISION=429305c8a5b3de597243d919a07e4e6bdcd00ef7"
set "SCRIPT_DIR=%~dp0"
set "OUTDIR=%SCRIPT_DIR%checkpoints\sam3"

REM Optional arg: fp32 (default) | fp16 | clean
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=fp32"

if /I "%VARIANT%"=="clean" (
  echo [INFO] Cleaning "%OUTDIR%" ...
  if exist "%OUTDIR%" rmdir /s /q "%OUTDIR%"
  echo [OK] Cleaned.
  pause
  exit /b 0
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "ENC=onnx/vision_encoder.onnx"
set "ENC_DATA=onnx/vision_encoder.onnx_data"
set "DEC=onnx/prompt_encoder_mask_decoder.onnx"
set "DEC_DATA=onnx/prompt_encoder_mask_decoder.onnx_data"

if /I "%VARIANT%"=="fp16" (
  set "ENC=onnx/vision_encoder_fp16.onnx"
  set "ENC_DATA=onnx/vision_encoder_fp16.onnx_data"
  set "DEC=onnx/prompt_encoder_mask_decoder_fp16.onnx"
  set "DEC_DATA=onnx/prompt_encoder_mask_decoder_fp16.onnx_data"
) else if /I "%VARIANT%"=="fp32" (
  REM keep defaults
) else (
  echo [WARN] Unknown variant "%VARIANT%". Using fp32.
  set "VARIANT=fp32"
)

echo [INFO] Variant: %VARIANT%
echo [INFO] HF revision: %HF_REVISION%
echo [INFO] Downloading to "%OUTDIR%" ...

python -c "from huggingface_hub import hf_hub_download; import os; repo=r'%REPO_ID%'; rev=r'%HF_REVISION%'; out=r'%OUTDIR%'; files=[r'%ENC%', r'%ENC_DATA%', r'%DEC%', r'%DEC_DATA%']; os.makedirs(out, exist_ok=True); [hf_hub_download(repo_id=repo, filename=f, revision=rev, local_dir=out) for f in files]; print('[OK] Downloaded from ' + repo + '@' + rev + ':\n  ' + '\n  '.join(files))"

if errorlevel 1 (
  echo [ERROR] Download failed. Make sure the venv is activated and huggingface_hub is installed.
  exit /b 1
)

echo [OK] Done.
