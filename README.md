# sam3-onnx-cpp

SAM3 ONNX experiments and wrappers built around the SAM3 tracker/image model.

This repo now contains both Python and C++ paths:

- Python ONNX demos for image segmentation and video tracking.
- A C++ `Segment` executable for image segmentation and video tracking through ONNX Runtime.
- An exporter for tracker-specific video ONNX modules built from a local `sam3` checkout.
- Native SAM3 reference scripts for checking parity against the ONNX path.
- Benchmark helpers and CPU quantization experiments.

The image encoder is still the shipped Hugging Face ONNX encoder. The repo exports and wraps the video tracker-specific modules around that encoder.

## Status Snapshot

Current working state:

- C++ image path works on CPU and CUDA.
- C++ video path works on CPU and CUDA.
- The video path uses the downloaded `vision_encoder*.onnx` plus the locally exported tracker bundle.
- CPU auto mode prefers the available int8 image encoder/decoder artifacts.
- CUDA auto mode prefers fp16 image/tracker artifacts.
- The video smoke test should be kept small on CPU, for example `--max_frames 2` or `--max_frames 3`.

Performance expectations:

- CPU inference is supported but can be slow because the SAM3 vision encoder is large.
- CUDA fp16 inference is the recommended path for interactive video use.
- The first annotated video frame may report `Enc: 0 ms` when encoder outputs were already cached during preview. Propagated frames pay the encoder cost.
- ONNX Runtime may print CUDA graph-optimization warnings with the current exported graphs.

## What Comes From Hugging Face vs This Repo

### Downloaded from Hugging Face

The image ONNX pair is downloaded from:

- `onnx-community/sam3-tracker-ONNX`

Files fetched by `fetch_onnx_models.bat` / `fetch_onnx_models.sh`:

- `onnx/vision_encoder.onnx`
- `onnx/vision_encoder.onnx_data`
- `onnx/vision_encoder_fp16.onnx`
- `onnx/vision_encoder_fp16.onnx_data`
- `onnx/prompt_encoder_mask_decoder.onnx`
- `onnx/prompt_encoder_mask_decoder.onnx_data`
- `onnx/prompt_encoder_mask_decoder_fp16.onnx`
- `onnx/prompt_encoder_mask_decoder_fp16.onnx_data`

These are the models used by the image-only ONNX demo.

### Exported by this repo

The exporter in `export/` builds tracker-specific ONNX modules from a local SAM3 checkout:

- `checkpoints/sam3/video_onnx/image_decoder_single.onnx`
- `checkpoints/sam3/video_onnx/image_decoder_single_fp16.onnx`
- `checkpoints/sam3/video_onnx/memory_attention_single.onnx`
- `checkpoints/sam3/video_onnx/memory_attention_single_fp16.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder_single.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder_single_fp16.onnx`
- `checkpoints/sam3/video_onnx/video_constants_single.npz`
- `checkpoints/sam3/video_onnx/video_constants_single_fp16.npz`
- `checkpoints/sam3/video_onnx/image_decoder_multi.onnx`
- `checkpoints/sam3/video_onnx/image_decoder_multi_fp16.onnx`
- `checkpoints/sam3/video_onnx/memory_attention_multi.onnx`
- `checkpoints/sam3/video_onnx/memory_attention_multi_fp16.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder_multi.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder_multi_fp16.onnx`
- `checkpoints/sam3/video_onnx/video_constants_multi.npz`
- `checkpoints/sam3/video_onnx/video_constants_multi_fp16.npz`

Important limitation:

- This repo does not currently export its own image encoder ONNX.
- The video path reuses the downloaded `vision_encoder*.onnx` model and combines it with the locally exported tracker modules above.

## Current Capabilities

### Image ONNX path

- Interactive segmentation on a single image.
- Prompt types: positive / negative seed points and bounding boxes.
- Uses ONNX Runtime with CPU, CUDA, or TensorRT execution providers when available.

### Video ONNX path

- Tracker propagation over video frames.
- Supports single-frame and multi-frame prompt annotations.
- Uses an internal `single` graph for one annotation and a `multi` graph for multi-annotation clips.
- There are no user-facing preset names anymore.
- On CUDA/TensorRT, the runtime automatically prefers the fp16 tracker bundle and warms the kernels on startup.
- Uses the downloaded vision encoder plus repo-exported tracker modules.

### Native reference path

- Native SAM3 image demo.
- Native-vs-ONNX comparison and benchmark scripts for the video tracker path.

## Repository Layout

```text
sam3-onnx-cpp/
|-- cpp/
|   |-- CMakeLists.txt
|   `-- src/
|-- export/
|   |-- onnx_export.py
|   `-- src/
|-- python/
|   |-- onnx_test_image.py
|   |-- onnx_test_video.py
|   |-- inspect_onnx_io.py
|   |-- compare_native_vs_onnx.py
|   |-- benchmark_onnx_default.py
|   |-- sweep_onnx_mem_frames.py
|   |-- sweep_onnx_obj_ptrs.py
|   `-- api_test_image.py
|-- checkpoints/
|   `-- sam3/
|       |-- onnx/
|       `-- video_onnx/
|-- fetch_onnx_models.bat
|-- fetch_onnx_models.sh
`-- README.md
```

## Preprocessing Notes

The downloaded image ONNX export expects:

- `pixel_values` resized directly to `1008x1008`
- No padding
- RGB input
- Values scaled by `1/255`
- Mean/std normalization with `0.5`

Point and box coordinates are scaled independently with `scale_x` and `scale_y` to match the resized image.

## Environment Setup

Use one environment for everything in this repo: ONNX demos, tracker export, and native SAM3 comparison.

```powershell
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
pip install torch onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

If you also have `conda` active in the same shell, run `conda deactivate` first so the venv owns the DLL search path cleanly on Windows.

You also need a local `sam3` repository next to this repo by default:

```text
../sam3
```

or pass `--sam3-repo` explicitly.

## Quick Deploy

This is the shortest Windows-first path to get the project running from a fresh checkout.

### ONNX-only deploy

Use the same `sam3_env` environment even if you only want the ONNX image demo or the ONNX video tracker.

1. Clone this repo.
2. Create and activate the ONNX environment:

```powershell
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
pip install torch onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

3. Download the Hugging Face encoder/image ONNX files:

```powershell
.\fetch_onnx_models.bat fp32
.\fetch_onnx_models.bat fp16
```

4. Make sure these files exist under `checkpoints/sam3/onnx`:

- `vision_encoder.onnx`
- `vision_encoder.onnx_data`
- `vision_encoder_fp16.onnx`
- `vision_encoder_fp16.onnx_data`
- `prompt_encoder_mask_decoder.onnx`
- `prompt_encoder_mask_decoder.onnx_data`
- `prompt_encoder_mask_decoder_fp16.onnx`
- `prompt_encoder_mask_decoder_fp16.onnx_data`

5. For video tracking, also make sure the tracker bundle exists under `checkpoints/sam3/video_onnx`:

- `image_decoder_single.onnx`
- `memory_attention_single.onnx`
- `memory_encoder_single.onnx`
- `video_constants_single.npz`
- `image_decoder_multi.onnx`
- `memory_attention_multi.onnx`
- `memory_encoder_multi.onnx`
- `video_constants_multi.npz`

The runtime will automatically pick the fp16 tracker files when CUDA/TensorRT is available and the `_fp16` bundle exists.

6. Run the image demo:

```powershell
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points
```

7. Or run the video tracker:

```powershell
.\sam3_env\Scripts\python.exe python\onnx_test_video.py --video "C:\path\to\video.mp4" --prompt seed_points
```

### Full deploy with export and native comparison

Use this if you want the complete repo workflow, including exporting tracker graphs from a local SAM3 checkout and benchmarking against native PyTorch SAM3.

1. Clone this repo.
2. Clone `sam3` next to it so the default layout is:

```text
../sam3
../sam3-onnx-cpp
```

3. Create and activate the export/native environment:

```powershell
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
pip install torch onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

4. Download the Hugging Face encoder/image ONNX files:

```powershell
.\fetch_onnx_models.bat fp32
.\fetch_onnx_models.bat fp16
```

5. Export the tracker bundle:

```powershell
.\sam3_env\Scripts\python.exe export\onnx_export.py `
  --sam3-repo "..\sam3" `
  --load-from-hf
```

6. Verify that both directories now exist and are populated:

- `checkpoints/sam3/onnx`
- `checkpoints/sam3/video_onnx`

7. Run the ONNX video demo:

```powershell
.\sam3_env\Scripts\python.exe python\onnx_test_video.py --video "C:\path\to\video.mp4"
```

8. Run the native-vs-ONNX comparison:

```powershell
.\sam3_env\Scripts\python.exe python\compare_native_vs_onnx.py `
  --video "C:\path\to\video.mp4" `
  --sam3_repo "..\sam3" `
  --checkpoint "C:\path\to\sam3.pt"
```

## Optional GPU Setup

### Modern NVIDIA stack

```powershell
pip uninstall -y onnxruntime
pip install onnx "onnxruntime-gpu[cuda,cudnn]" huggingface_hub pillow opencv-python pyqt5 numpy
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You want to see `CUDAExecutionProvider`.

### Pascal / Volta-friendly stack

```powershell
pip uninstall -y onnxruntime onnxruntime-gpu `
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 `
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-nvjitlink-cu12 nvidia-cudnn-cu12

pip install onnx "onnxruntime-gpu==1.22.0" huggingface_hub pillow opencv-python pyqt5 numpy

pip install `
  "nvidia-cuda-runtime-cu12==12.5.82" `
  "nvidia-cuda-nvrtc-cu12==12.5.82" `
  "nvidia-cublas-cu12==12.5.3.2" `
  "nvidia-cufft-cu12==11.4.1.4" `
  "nvidia-curand-cu12==10.3.10.19" `
  "nvidia-nvjitlink-cu12==12.5.82" `
  "nvidia-cudnn-cu12==9.10.2.21"

python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Build the C++ Runtime

The C++ target is `Segment`. It wraps both the image and video ONNX paths.

Configure with OpenCV and ONNX Runtime:

```powershell
cmake -S cpp -B cpp\build_msvc -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR="C:\path\to\opencv\build" `
  -DONNXRUNTIME_DIR="C:\path\to\onnxruntime-win-x64-gpu-1.22.0"
```

Build:

```powershell
cmake --build cpp\build_msvc --config Release --target Segment -- /m:1
```

The executable is written to:

```text
cpp/build_msvc/bin/Release/Segment.exe
```

### C++ Image Smoke Tests

CPU:

```powershell
.\cpp\build_msvc\bin\Release\Segment.exe `
  --onnx_test_image `
  --device cpu `
  --threads 8 `
  --image tmp\sam3_smoke.png `
  --box 90,55,230,185 `
  --save_overlay tmp\image_cpu.png `
  --no_gui
```

CUDA:

```powershell
.\cpp\build_msvc\bin\Release\Segment.exe `
  --onnx_test_image `
  --device cuda `
  --threads 4 `
  --image tmp\sam3_smoke.png `
  --box 90,55,230,185 `
  --save_overlay tmp\image_gpu.png `
  --no_gui
```

### C++ Video Smoke Tests

CPU, keep this short:

```powershell
.\cpp\build_msvc\bin\Release\Segment.exe `
  --onnx_test_video `
  --device cpu `
  --threads 8 `
  --video tmp\sam3_smoke.avi `
  --box 90,55,230,185 `
  --max_frames 3 `
  --output tmp\video_cpu.avi
```

CUDA:

```powershell
.\cpp\build_msvc\bin\Release\Segment.exe `
  --onnx_test_video `
  --device cuda `
  --threads 4 `
  --video tmp\sam3_smoke.avi `
  --box 90,55,230,185 `
  --max_frames 3 `
  --output tmp\video_gpu.avi
```

Useful C++ options:

- `--device cpu|cuda|cuda:N`
- `--threads N`
- `--points x,y,label;...`
- `--box x1,y1,x2,y2`
- `--max_frames N`
- `--output path`
- `--save_overlay path`
- `--no_gui` for noninteractive image runs

## Download the Hugging Face ONNX Models

You can keep both FP32 and FP16 variants in the same folder.

### Windows

```powershell
.\fetch_onnx_models.bat fp32
.\fetch_onnx_models.bat fp16
```

### macOS / Linux

```bash
chmod +x fetch_onnx_models.sh
./fetch_onnx_models.sh fp32
./fetch_onnx_models.sh fp16
```

To remove the downloaded models:

```powershell
.\fetch_onnx_models.bat clean
```

## Inspect ONNX I/O

```powershell
python python\inspect_onnx_io.py
```

Force a specific image precision:

```powershell
$env:SAM3_ONNX_VARIANT="fp16"
python python\inspect_onnx_io.py
```

## Run SAM3 ONNX on a Particular Image

This is the simplest way to run the ONNX image path on one exact file:

```powershell
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points
```

Or with a box prompt:

```powershell
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt bounding_box
```

If you omit `--image`, the script opens a file picker instead.

### Controls

#### Seed points mode

- Left click: positive point
- Right click: negative point
- Middle click: clear points
- `Esc`: quit

#### Bounding box mode

- Drag left mouse button: draw box
- Right click or double click: clear box
- `Esc`: quit

### Useful options

Disable ORT graph optimizations:

```powershell
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points --safe
```

Force CUDA and prefer the FP16 image ONNX files:

```powershell
$env:SAM3_ORT_ACCEL="cuda"
$env:SAM3_ONNX_VARIANT="fp16"
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points
```

### CPU Performance Notes

The C++ runtime prefers CPU int8 image artifacts when they are available. It checks these paths before falling back to fp32:

- `checkpoints/sam3/onnx/vision_encoder.int8.onnx`
- `checkpoints/sam3/onnx/bench_cpu/vision_encoder.int8.matmul_gather.onnx`
- `checkpoints/sam3/onnx/bench_cpu/vision_encoder.int8.matmul_gather_pre.onnx`
- `checkpoints/sam3/onnx/bench_cpu/vision_encoder.int8.matmul.onnx`
- `checkpoints/sam3/onnx/bench_cpu/prompt_encoder_mask_decoder.int8.matmul_gemm.onnx`

To force the original fp32 image models instead:

```powershell
$env:SAM3_ORT_ENCODER_VARIANT="fp32"
$env:SAM3_ORT_DECODER_VARIANT="fp32"
.\cpp\build_msvc\bin\Release\Segment.exe --onnx_test_image --device cpu --image "C:\path\to\image.jpg" --box 100,80,420,350 --no_gui
```

The CPU bottleneck is the SAM3 vision encoder, not the prompt decoder or the C++ wrapper. The local SAM3 source builds the visual backbone as a large ViT:

- Input resolution: `1008x1008`
- Patch size: `14`
- Token grid: `72x72`, or `5184` tokens
- Width: `1024`
- Depth: `32`
- Heads: `16`
- Global attention blocks: `7, 15, 23, 31`

That is why CUDA is expected to be significantly faster than CPU for this model.

Practical CPU options:

- Use the int8 artifacts for smoke tests and CPU fallback.
- Try a few `--threads N` values for the target machine and benchmark more than one run.
- Keep CPU video smoke tests to `2` or `3` frames.
- For real video throughput, use CUDA or TensorRT.

Further CPU acceleration would require model-level changes rather than wrapper-only changes.

## Run the ONNX Video Demo

The video path expects:

- `checkpoints/sam3/video_onnx/image_decoder_single.onnx`
- `checkpoints/sam3/video_onnx/memory_attention_single.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder_single.onnx`
- `checkpoints/sam3/video_onnx/video_constants_single.npz`
- `checkpoints/sam3/video_onnx/image_decoder_multi.onnx`
- `checkpoints/sam3/video_onnx/memory_attention_multi.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder_multi.onnx`
- `checkpoints/sam3/video_onnx/video_constants_multi.npz`
- A complete `vision_encoder*.onnx` plus matching `.onnx_data` under `checkpoints/sam3/onnx`

Example with interactive first-frame points:

```powershell
python python\onnx_test_video.py --video "C:\path\to\video.mp4" --prompt seed_points
```

Example with a noninteractive box:

```powershell
python python\onnx_test_video.py --video "C:\path\to\video.mp4" --box "120,80,520,430"
```

Default runtime behavior:

- Single annotation: uses the internal `single` graph and defaults to `--max_mem_frames 2`.
- Multi-annotation: uses the internal `multi` graph and defaults to `--max_mem_frames 4`.
- `--max_obj_ptrs` defaults to `16`.
- `SAM3_ORT_TRACKER_PRECISION=auto` prefers fp16 tracker graphs on CUDA/TensorRT and falls back to fp32 when needed.
- `SAM3_ORT_WARMUP=auto` pre-warms the ONNX tracker kernels so the first real frame does not pay the full cold-start cost.

## Export the Tracker ONNX Modules

The exporter requires:

- A local SAM3 checkout, defaulting to `../sam3`
- A PyTorch-capable environment
- Either `--checkpoint` or `--load-from-hf`

Example:

```powershell
.\sam3_env\Scripts\python.exe export\onnx_export.py `
  --sam3-repo "..\sam3" `
  --load-from-hf
```

Or with a local checkpoint:

```powershell
.\sam3_env\Scripts\python.exe export\onnx_export.py `
  --sam3-repo "..\sam3" `
  --checkpoint "C:\path\to\sam3.pt"
```

By default this writes to:

```text
checkpoints/sam3/video_onnx
```

The exporter writes two internal tracker bundles:

- `single`: static 2-slot memory graph for the common one-annotation path
- `multi`: static 4-slot memory graph for multi-annotation clips

It emits both `fp32` and `fp16` tracker graphs by default.

## Run the Native Image Reference Demo

This is useful for checking native SAM3 behavior outside ONNX:

```powershell
.\sam3_env\Scripts\python.exe python\api_test_image.py --prompt seed_points
```

## Compare Native vs ONNX Video Tracking

```powershell
.\sam3_env\Scripts\python.exe python\compare_native_vs_onnx.py `
  --video "C:\path\to\video.mp4" `
  --sam3_repo "..\sam3" `
  --checkpoint "C:\path\to\sam3.pt" `
  --prompt seed_points
```

## Benchmark The Default ONNX Runtime

```powershell
.\sam3_env\Scripts\python.exe python\benchmark_onnx_default.py `
  --video "C:\path\to\video.mp4" `
  --sam3_repo "..\sam3" `
  --checkpoint "C:\path\to\sam3.pt"
```

## Summary

If you only want to segment one image with ONNX:

1. Create and activate `sam3_env`.
2. Install the ONNX Runtime dependencies.
3. Download the Hugging Face ONNX files with `fetch_onnx_models.bat`.
4. Run:

```powershell
.\sam3_env\Scripts\python.exe python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points
```

That is the shortest path for running SAM3 ONNX on a particular image in this repo.
