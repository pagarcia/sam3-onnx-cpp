# sam3-onnx-cpp

SAM3 ONNX experiments and wrappers built around the SAM3 tracker/image model.

This repo is currently Python-first. It contains:

- An ONNX Runtime image demo that uses pre-exported Hugging Face ONNX models.
- An exporter for tracker-specific video ONNX modules built from a local `sam3` checkout.
- Video demos, comparison scripts, and benchmark helpers for the exported tracker path.
- Native SAM3 reference scripts for checking parity against the ONNX path.

There is no `cpp/` implementation checked into this checkout right now, despite the repository name.

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

- `checkpoints/sam3/video_onnx/image_decoder.onnx`
- `checkpoints/sam3/video_onnx/memory_attention.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder.onnx`
- `checkpoints/sam3/video_onnx/video_constants.npz`

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
- First-frame prompt can be seed points or a box.
- Uses the downloaded vision encoder plus repo-exported tracker modules.

### Native reference path

- Native SAM3 image demo.
- Native-vs-ONNX comparison and benchmark scripts for the video tracker path.

## Repository Layout

```text
sam3-onnx-cpp/
|-- export/
|   |-- onnx_export.py
|   `-- src/
|-- python/
|   |-- onnx_test_image.py
|   |-- onnx_test_video.py
|   |-- inspect_onnx_io.py
|   |-- compare_native_vs_onnx.py
|   |-- benchmark_onnx_presets.py
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

There are really two workflows in this repo.

### 1. ONNX Runtime only

Use this if you only want to run the downloaded image ONNX demo or the exported video ONNX demo.

```powershell
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
pip install onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

### 2. Export / native comparison tooling

Use this if you want to export tracker modules or run native PyTorch SAM3 comparisons. This environment needs `torch` plus access to a local SAM3 checkout.

Example:

```powershell
python -m venv sam3_api_env
.\sam3_api_env\Scripts\Activate.ps1
pip install torch onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

You also need a local `sam3` repository next to this repo by default:

```text
../sam3
```

or pass `--sam3-repo` explicitly.

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

Force a specific variant:

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

Force CUDA and prefer the FP16 ONNX files:

```powershell
$env:SAM3_ORT_ACCEL="cuda"
$env:SAM3_ONNX_VARIANT="fp16"
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points
```

## Run the ONNX Video Demo

The video path expects:

- `checkpoints/sam3/video_onnx/image_decoder.onnx`
- `checkpoints/sam3/video_onnx/memory_attention.onnx`
- `checkpoints/sam3/video_onnx/memory_encoder.onnx`
- `checkpoints/sam3/video_onnx/video_constants.npz`
- A complete `vision_encoder*.onnx` plus matching `.onnx_data` under `checkpoints/sam3/onnx`

Example with interactive first-frame points:

```powershell
python python\onnx_test_video.py --video "C:\path\to\video.mp4" --prompt seed_points
```

Example with a noninteractive box:

```powershell
python python\onnx_test_video.py --video "C:\path\to\video.mp4" --box "120,80,520,430"
```

Fast preset controls:

- `--max_mem_frames 2`
- `--max_obj_ptrs 16`

Quality-oriented spatial memory:

```powershell
python python\onnx_test_video.py --video "C:\path\to\video.mp4" --prompt seed_points --max_mem_frames 7 --max_obj_ptrs 16
```

## Export the Tracker ONNX Modules

The exporter requires:

- A local SAM3 checkout, defaulting to `../sam3`
- A PyTorch-capable environment
- Either `--checkpoint` or `--load-from-hf`

Example:

```powershell
.\sam3_api_env\Scripts\python.exe export\onnx_export.py `
  --sam3-repo "..\sam3" `
  --load-from-hf
```

Or with a local checkpoint:

```powershell
.\sam3_api_env\Scripts\python.exe export\onnx_export.py `
  --sam3-repo "..\sam3" `
  --checkpoint "C:\path\to\sam3.pt"
```

By default this writes to:

```text
checkpoints/sam3/video_onnx
```

## Run the Native Image Reference Demo

This is useful for checking native SAM3 behavior outside ONNX:

```powershell
.\sam3_api_env\Scripts\python.exe python\api_test_image.py --prompt seed_points
```

## Compare Native vs ONNX Video Tracking

```powershell
.\sam3_api_env\Scripts\python.exe python\compare_native_vs_onnx.py `
  --video "C:\path\to\video.mp4" `
  --sam3_repo "..\sam3" `
  --checkpoint "C:\path\to\sam3.pt" `
  --prompt seed_points
```

## Benchmark ONNX Presets

```powershell
.\sam3_api_env\Scripts\python.exe python\benchmark_onnx_presets.py `
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
python python\onnx_test_image.py --image "C:\path\to\image.jpg" --prompt seed_points
```

That is the shortest path for running SAM3 ONNX on a particular image in this repo.
