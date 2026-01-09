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

> Note on resolution / preprocessing:
> This ONNX export expects `pixel_values` resized directly to **1008×1008** (no padding).
> Point/box coordinates are scaled with **scale_x** and **scale_y** accordingly.

---

## Windows Setup & Execution

### 1) Create a Python Virtual Environment

In the repository root:
```bash
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
````

### 2) Install Dependencies

#### 2.1 CPU-only (default)

```bash
pip install onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

#### 2.2 NVIDIA GPU (recommended) — CUDA/cuDNN via pip

This installs ONNX Runtime GPU plus CUDA + cuDNN runtime DLLs (no full CUDA toolkit required):

```bash
pip uninstall -y onnxruntime
pip install onnx "onnxruntime-gpu[cuda,cudnn]" huggingface_hub pillow opencv-python pyqt5 numpy
```

Verify CUDA provider shows up:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You want to see `CUDAExecutionProvider` in the list.

### 3) Download the ONNX Models

#### 3.1 CPU weights (FP32, bigger download)

```bash
.\fetch_onnx_models.bat
```

#### 3.2 GPU-friendly weights (FP16, recommended for CUDA)

```bash
.\fetch_onnx_models.bat fp16
```

This downloads into:
`checkpoints/sam3/onnx/`

### 4) Sanity-check ONNX model I/O

```bash
python python\inspect_onnx_io.py
```

Expected highlights:

* Encoder input: `pixel_values` `[B,3,1008,1008]`
* Decoder inputs: `input_points`, `input_labels`, `input_boxes`, plus `image_embeddings.0/.1/.2`
* Outputs: `iou_scores`, `pred_masks`, `object_score_logits`

### 5) Run Python Image Demo

Seed points:

```bash
python python\onnx_test_image.py --prompt seed_points
```

Bounding box:

```bash
python python\onnx_test_image.py --prompt bounding_box
```

Optional:

* Disable ORT graph optimizations (more conservative):

```bash
python python\onnx_test_image.py --prompt seed_points --safe
```

* Force CUDA:

```bash
$env:SAM3_ORT_ACCEL="cuda"
python python\onnx_test_image.py --prompt seed_points
```

* Force a specific model variant:

```bash
$env:SAM3_ONNX_VARIANT="fp16"   # fp16 | fp32
python python\onnx_test_image.py --prompt seed_points
```

---

## macOS Setup & Execution

### 1) Create a Python Virtual Environment

```bash
python -m venv sam3_env
source sam3_env/bin/activate
```

### 2) Install Dependencies (CPU-only recommended)

```bash
pip install onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

### 3) Download the ONNX Models

CPU (FP32):

```bash
chmod +x fetch_onnx_models.sh
./fetch_onnx_models.sh
```

GPU-friendly (FP16) (useful on platforms that support CUDA EP):

```bash
./fetch_onnx_models.sh fp16
```

### 4) Sanity-check ONNX model I/O

```bash
python python/inspect_onnx_io.py
```

### 5) Run Python Image Demo

```bash
python python/onnx_test_image.py --prompt seed_points
python python/onnx_test_image.py --prompt bounding_box
```

---

## Project Structure (initial)

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