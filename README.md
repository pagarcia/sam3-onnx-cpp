# sam3-onnx-cpp

**Segment Anything Model 3 (SAM3) C++ ONNX Wrapper (Image / 2D)**

This repository provides a C++ wrapper for SAM3 **Promptable Visual Segmentation (PVS)** (interactive points / boxes) using ONNX Runtime.

### What this repo targets (first milestone)
- **Image-only (2D)** segmentation (no video/tracking yet)
- Prompts:
  - **Seed points** (positive/negative clicks)
  - **Bounding boxes**
- Uses the ONNX split:
  - `vision_encoder.onnx` (+ `.onnx_data`)
  - `prompt_encoder_mask_decoder.onnx` (+ `.onnx_data`)

We start from the **pre-exported ONNX models** published as:
- `onnx-community/sam3-tracker-ONNX`

> Note on resolution:
> The tracker config is built around **image_size=1008** (72×72 tokens with patch size 14).
> In practice: you should resize input images to 1008×1008 for best results.

---

## Windows Setup & Execution

### 1) Create a Python Virtual Environment

In the repository root:
```bash
python -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
````

### 2) Install Dependencies

#### 2.1 CPU-only

```bash
pip install onnx onnxruntime huggingface_hub pillow opencv-python pyqt5 numpy
```

#### 2.2 NVIDIA GPU (optional)

```
pip install onnx onnxruntime-gpu huggingface_hub pillow opencv-python pyqt5 numpy
```

### 3) Download the Pre-exported ONNX Models

Run:

```bash
.\fetch_onnx_models.bat
```

This downloads:

* `checkpoints/sam3/onnx/vision_encoder.onnx`
* `checkpoints/sam3/onnx/vision_encoder.onnx_data`
* `checkpoints/sam3/onnx/prompt_encoder_mask_decoder.onnx`
* `checkpoints/sam3/onnx/prompt_encoder_mask_decoder.onnx_data`

### 4) Sanity-check ONNX model I/O

Run:

```bash
python python\inspect_onnx_io.py
```

Expected highlights:

* `vision_encoder.onnx` input: `pixel_values` `[B,3,1008,1008]`
* `prompt_encoder_mask_decoder.onnx` inputs include:

  * `input_points`, `input_labels`, `input_boxes`
  * `image_embeddings.0/.1/.2`
* outputs include:

  * `iou_scores`, `pred_masks`, `object_score_logits`

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

* Force CUDA (if you installed onnxruntime-gpu):

```bash
$env:SAM3_ORT_ACCEL="cuda"
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

### 3) Download the Pre-exported ONNX Models

```bash
chmod +x fetch_onnx_models.sh
./fetch_onnx_models.sh
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

Optional (CUDA on platforms that support it):

```bash
export SAM3_ORT_ACCEL=cuda
python python/onnx_test_image.py --prompt seed_points
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