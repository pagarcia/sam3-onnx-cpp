You’re right to question it — the **“~12.9 + cuDNN ~9.14”** recipe is a *modern* stack (good for Ampere/Ada), **not** the “legacy / Pascal” fix. Your crash (`no kernel image is available`) is exactly what happens when a cuDNN build no longer ships kernels for your GPU; NVIDIA’s current cuDNN backend release notes explicitly say **Maxwell/Pascal/Volta are no longer supported**. ([NVIDIA Docs][1])

Also, `onnxruntime-gpu==1.22.1` doesn’t exist on PyPI (only `1.22.0` in that series). ([PyPI][2])

Below are **full drop-in files** updated to:

* Use **onnxruntime-gpu==1.22.0** for the stable/Pascal-friendly stack (not 1.22.1) ([PyPI][2])
* Keep a **modern GPU** path using `onnxruntime-gpu[cuda,cudnn]` per ORT docs ([ONNX Runtime][3])
* Explain why Pascal may fail with the newest cuDNN ([NVIDIA Docs][1])
* Fix print statements to show **requested vs effective** model variant and prevent “fp32” while loading fp16 confusion
* Fix fetch scripts so they **don’t delete** the folder (so fp32 and fp16 can coexist), and print newlines correctly

---

# ✅ `README.md` (full)

````md
# sam3-onnx-cpp

**Segment Anything Model 3 (SAM3) C++ ONNX Wrapper (Image / 2D)**

This repository provides a C++ wrapper for SAM3 **Promptable Visual Segmentation (PVS)** (interactive points / boxes) using ONNX Runtime.

### What this repo targets (first milestone)
- **Image-only (2D)** segmentation (no video/tracking yet)
- Prompts:
  - **Seed points** (positive/negative clicks)
  - **Bounding boxes**
- Uses the ONNX split:
  - `vision_encoder*.onnx` (+ `.onnx_data`)
  - `prompt_encoder_mask_decoder*.onnx` (+ `.onnx_data`)

We start from the **pre-exported ONNX models** published as:
- `onnx-community/sam3-tracker-ONNX`

> Note on preprocessing:
> This ONNX export expects `pixel_values` resized directly to **1008×1008** (no padding).
> Point/box coordinates are scaled with **scale_x** and **scale_y** accordingly (see `python/onnx_test_utils.py`).

---

## Windows Setup & Execution

### 1) Create a Python Virtual Environment

In the repository root:

```powershell
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
````

### 2) Install Dependencies

#### 2.1 CPU-only (default)

```powershell
pip install onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

#### 2.2 NVIDIA GPU – Modern stack (Turing/Ampere/Ada and newer)

ONNX Runtime documents that you can install CUDA + cuDNN runtime DLLs via pip extras: ([ONNX Runtime][3])

```powershell
pip uninstall -y onnxruntime
pip install onnx "onnxruntime-gpu[cuda,cudnn]" huggingface_hub pillow opencv-python pyqt5 numpy
```

Verify CUDA EP is available:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You want to see `CUDAExecutionProvider`.

> If you have Pascal/Volta and see errors like `no kernel image is available`,
> it’s usually because newer cuDNN releases don’t support pre-Turing GPUs anymore. ([NVIDIA Docs][1])
> Use the “Stable (Pascal/Volta)” stack below.

#### 2.3 NVIDIA GPU – Stable stack (Pascal/Volta-friendly)

PyPI provides `onnxruntime-gpu==1.22.0` (note: **there is no 1.22.1**). ([PyPI][2])
Install ORT 1.22.0 + a pinned CUDA 12.5 + cuDNN 9.10 runtime set (matches the SAM2 proven combo):

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
```

Verify:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

---

### 3) Download the Pre-exported ONNX Models

You can keep both FP32 and FP16 variants in the same folder.

#### 3.1 Download FP32 (CPU-friendly, larger)

```powershell
.\fetch_onnx_models.bat fp32
```

#### 3.2 Download FP16 (GPU-friendly, recommended on CUDA)

```powershell
.\fetch_onnx_models.bat fp16
```

Optional: delete all downloaded models:

```powershell
.\fetch_onnx_models.bat clean
```

---

### 4) Sanity-check ONNX model I/O

```powershell
python python\inspect_onnx_io.py
```

You can force a variant:

```powershell
$env:SAM3_ONNX_VARIANT="fp16"
python python\inspect_onnx_io.py
```

---

### 5) Run Python Image Demo

Seed points:

```powershell
python python\onnx_test_image.py --prompt seed_points
```

Bounding box:

```powershell
python python\onnx_test_image.py --prompt bounding_box
```

Optional:

* Disable ORT graph optimizations:

```powershell
python python\onnx_test_image.py --prompt seed_points --safe
```

* Force CUDA + FP16:

```powershell
$env:SAM3_ORT_ACCEL="cuda"
$env:SAM3_ONNX_VARIANT="fp16"
python python\onnx_test_image.py --prompt seed_points
```

---

## macOS Setup & Execution (CPU)

```bash
python -m venv sam3_env
source sam3_env/bin/activate
pip install onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
chmod +x fetch_onnx_models.sh
./fetch_onnx_models.sh fp32
python python/onnx_test_image.py --prompt seed_points
```

---

## Project Structure

```
sam3-onnx-cpp/
├── export/                 # (optional later) exporter scripts
├── python/
│   ├── inspect_onnx_io.py
│   ├── onnx_test_image.py
│   └── onnx_test_utils.py
├── cpp/                    # C++ wrapper + tests
├── checkpoints/
│   └── sam3/
│       └── onnx/           # downloaded ONNX files live here
├── fetch_onnx_models.bat
├── fetch_onnx_models.sh
├── LICENSE
└── README.md
```