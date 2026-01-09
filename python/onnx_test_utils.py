# sam3-onnx-cpp/python/onnx_test_utils.py
"""
Utilities for SAM3-Tracker ONNX (image-only) testing.

This repo uses the ONNX split from:
  onnx-community/sam3-tracker-ONNX

Important behavior for THIS ONNX:
- The encoder expects pixel_values shaped [B, 3, 1008, 1008]
- The preprocessor config for this export resizes directly to 1008x1008
  (no aspect-ratio padding).
- The decoder takes three prompt inputs:
    input_points: [B, 1, N, 2]
    input_labels: [B, 1, N]   (int64)
    input_boxes : [B, M, 4]
  and the image_embeddings.{0,1,2} from the encoder.

Very important:
- In seed-point mode, you MUST pass input_boxes as an EMPTY tensor [B,0,4]
  to avoid a dummy box affecting results.
- In bbox mode, you MUST pass points as EMPTY tensors [B,1,0,2] and [B,1,0].

Env toggles:
  SAM3_ORT_ACCEL = auto | cpu | cuda   (default: auto)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession

ACCEL = os.getenv("SAM3_ORT_ACCEL", "auto").lower()  # auto|cpu|cuda

# If onnxruntime-gpu is installed and CUDA EP is available, try preloading DLLs.
# This helps when CUDA/cuDNN are provided via pip packages or PyTorch.
try:
    if hasattr(ort, "preload_dlls") and "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()
except Exception:
    pass

# Normalize like the HF config for this ONNX export: mean=0.5 std=0.5 after rescale 1/255
_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
_STD  = np.array([0.5, 0.5, 0.5], np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Info / sessions
# ──────────────────────────────────────────────────────────────────────────────

def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] onnxruntime:", ort.__version__)
    print("[INFO] ORT providers (available):", ort.get_available_providers())


def set_cv2_threads(n: int = 1) -> None:
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass


def _cuda_providers(device_id: int = 0):
    return [
        ("CUDAExecutionProvider", {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
        }),
        "CPUExecutionProvider",
    ]


def make_session(path: str, tag: str = "model", safe: bool = False) -> InferenceSession:
    """
    safe=False: ORT_ENABLE_EXTENDED
    safe=True : ORT_DISABLE_ALL (more conservative)
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if safe else ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    so.intra_op_num_threads = max(1, (os.cpu_count() or 8) - 1)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    av = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]

    if ACCEL == "cpu":
        providers = ["CPUExecutionProvider"]
    elif ACCEL == "cuda":
        if "CUDAExecutionProvider" in av:
            providers = _cuda_providers()
        else:
            providers = ["CPUExecutionProvider"]
    else:  # auto
        if "CUDAExecutionProvider" in av:
            providers = _cuda_providers()
        else:
            providers = ["CPUExecutionProvider"]

    path = str(Path(path).resolve())
    print(f"[INFO] Loading {os.path.basename(path)} [{tag}] providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=list(providers))
    print("[INFO] Active providers:", sess.get_providers())
    return sess


def as_f32c(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


def _dtype_from_ort(ort_type: str):
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "uint8" in ort_type:
        return np.uint8
    return np.float32


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing (direct resize to 1008x1008)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PrepInfo:
    orig_hw: Tuple[int, int]   # (H, W) original
    target_size: int           # 1008
    scale_x: float             # target_size / W
    scale_y: float             # target_size / H


def preprocess_image_bgr(img_bgr: np.ndarray, target_size: int = 1008) -> Tuple[np.ndarray, PrepInfo]:
    """
    Preprocess for this ONNX export:
      - resize directly to (1008,1008)
      - RGB
      - rescale 1/255
      - normalize mean/std 0.5
    Returns:
      pixel_values: float32 [1,3,1008,1008]
      info: orig size + per-axis scales for prompt coordinates
    """
    H, W = img_bgr.shape[:2]
    scale_x = float(target_size) / float(W)
    scale_y = float(target_size) / float(H)

    img_resized = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    img_rgb = (img_rgb - _MEAN) / _STD

    pixel_values = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # [1,3,H,W]
    info = PrepInfo(orig_hw=(H, W), target_size=target_size, scale_x=scale_x, scale_y=scale_y)
    return as_f32c(pixel_values), info


# ──────────────────────────────────────────────────────────────────────────────
# Prompt prep
# ──────────────────────────────────────────────────────────────────────────────

def empty_points() -> Tuple[np.ndarray, np.ndarray]:
    pts = np.zeros((1, 1, 0, 2), dtype=np.float32)
    lbl = np.zeros((1, 1, 0), dtype=np.int64)
    return pts, lbl


def empty_boxes() -> np.ndarray:
    return np.zeros((1, 0, 4), dtype=np.float32)


def prepare_points(points_xy: Iterable[Tuple[int, int]],
                   labels: Iterable[int],
                   info: PrepInfo) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(list(points_xy), dtype=np.float32)
    lbl = np.asarray(list(labels), dtype=np.int64)

    if pts.size == 0:
        return empty_points()

    # Scale to 1008x1008 coordinates
    pts[:, 0] = pts[:, 0] * info.scale_x
    pts[:, 1] = pts[:, 1] * info.scale_y

    # [B, 1, N, 2] and [B, 1, N]
    pts = pts[None, None, :, :]
    lbl = lbl[None, None, :]
    return np.ascontiguousarray(pts), np.ascontiguousarray(lbl)


def prepare_boxes(rect_xyxy: Tuple[int, int, int, int], info: PrepInfo) -> np.ndarray:
    x1, y1, x2, y2 = rect_xyxy
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    box = np.array([[
        x1 * info.scale_x,
        y1 * info.scale_y,
        x2 * info.scale_x,
        y2 * info.scale_y,
    ]], dtype=np.float32)  # [1,4]

    return np.ascontiguousarray(box[None, :, :])  # [B, M(=1), 4]


# ──────────────────────────────────────────────────────────────────────────────
# Inference runners
# ──────────────────────────────────────────────────────────────────────────────

def run_encoder(sess_enc: InferenceSession, pixel_values: np.ndarray) -> Dict[str, np.ndarray]:
    inp_name = sess_enc.get_inputs()[0].name
    outs = sess_enc.run(None, {inp_name: as_f32c(pixel_values)})
    out_names = [o.name for o in sess_enc.get_outputs()]
    return dict(zip(out_names, outs))


def run_decoder(sess_dec: InferenceSession,
                enc_out: Dict[str, np.ndarray],
                input_points: Optional[np.ndarray] = None,
                input_labels: Optional[np.ndarray] = None,
                input_boxes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Decoder contract (from inspect_onnx_io.py):
      Inputs:
        input_points [B,1,N,2] float
        input_labels [B,1,N]   int64
        input_boxes  [B,M,4]   float
        image_embeddings.0/.1/.2
      Outputs:
        iou_scores [B, num_boxes_or_points, 3]
        pred_masks [B, num_boxes_or_points, num_masks, H, W]
        object_score_logits [B, num_boxes_or_points, 1]

    Use:
      - seed points: pass points/labels; pass input_boxes = empty_boxes()
      - box: pass input_boxes; pass points/labels = empty_points()
    """
    if input_points is None or input_labels is None:
        input_points, input_labels = empty_points()
    if input_boxes is None:
        input_boxes = empty_boxes()

    # Cast to the exact expected dtypes
    inps = sess_dec.get_inputs()
    dtype_points = _dtype_from_ort(inps[0].type)
    dtype_labels = _dtype_from_ort(inps[1].type)
    dtype_boxes  = _dtype_from_ort(inps[2].type)

    feed = {
        "input_points": np.ascontiguousarray(input_points.astype(dtype_points, copy=False)),
        "input_labels": np.ascontiguousarray(input_labels.astype(dtype_labels, copy=False)),
        "input_boxes":  np.ascontiguousarray(input_boxes.astype(dtype_boxes,  copy=False)),
        "image_embeddings.0": np.ascontiguousarray(enc_out["image_embeddings.0"]),
        "image_embeddings.1": np.ascontiguousarray(enc_out["image_embeddings.1"]),
        "image_embeddings.2": np.ascontiguousarray(enc_out["image_embeddings.2"]),
    }

    outs = sess_dec.run(None, feed)
    out_names = [o.name for o in sess_dec.get_outputs()]
    return dict(zip(out_names, outs))


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def pick_best_mask(pred_masks: np.ndarray, iou_scores: np.ndarray, which_prompt: int = 0) -> Tuple[np.ndarray, float]:
    """
    Expected shapes (from ONNX):
      pred_masks: [B, P, M, H, W]
      iou_scores: [B, P, 3]   (often 3 masks)
    We pick the best mask for batch=0 and prompt index `which_prompt`.
    """
    m = pred_masks[0, which_prompt]  # [M,H,W]
    s = iou_scores[0, which_prompt]  # [3]
    best = int(np.argmax(s))
    return m[best], float(s[best])


def postprocess_mask_to_original(mask_logits_2d: np.ndarray, info: PrepInfo) -> np.ndarray:
    """
    Since we resized directly to 1008x1008 (no padding), we:
      - upsample mask logits to 1008x1008
      - resize to original HxW
      - threshold at 0
    """
    H0, W0 = info.orig_hw
    T = info.target_size

    up = cv2.resize(mask_logits_2d, (T, T), interpolation=cv2.INTER_LINEAR)
    up0 = cv2.resize(up, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return (up0 > 0.0).astype(np.uint8) * 255


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def green_overlay(bgr: np.ndarray, mask255: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    fg = (mask255 > 0)
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)


def compute_display_base(img_bgr: np.ndarray, max_side: int = 1200) -> Tuple[np.ndarray, float]:
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(W, H))
    disp = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    return disp, scale