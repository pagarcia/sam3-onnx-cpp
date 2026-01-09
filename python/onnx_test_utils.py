# sam3-onnx-cpp/python/onnx_test_utils.py
"""
Utilities for SAM3-Tracker ONNX (image-only) testing.

We use a SAM-style preprocessing:
- Resize longest side to target_size=1008 (preserve aspect ratio)
- Pad to 1008x1008 (top-left anchored)
- RGB, rescale 1/255, normalize mean/std=0.5  => roughly [-1, 1]

Prompts:
- Points: labels 1 (pos), 0 (neg)
- Box: encoded as two points with labels 2 and 3 (top-left, bottom-right)

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

# Make pip-provided CUDA/cuDNN/TensorRT DLLs discoverable for this process (Windows)
try:
    ort.preload_dlls()
except Exception:
    pass

ACCEL = os.getenv("SAM3_ORT_ACCEL", "auto").lower()  # auto|cpu|cuda


# ──────────────────────────────────────────────────────────────────────────────
# Session helpers
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
    safe=True : ORT_DISABLE_ALL (more conservative; useful if some optimizations misbehave)
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

    if ACCEL == "cuda":
        if "CUDAExecutionProvider" in av:
            providers = _cuda_providers()
    elif ACCEL == "auto":
        if "CUDAExecutionProvider" in av:
            providers = _cuda_providers()

    path = str(Path(path).resolve())
    print(f"[INFO] Loading {os.path.basename(path)} [{tag}] providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=list(providers))
    print("[INFO] Active providers:", sess.get_providers())
    print("[INFO] Inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    return sess


def as_f32c(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing (resize longest side + pad) to 1008
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
_STD  = np.array([0.5, 0.5, 0.5], np.float32)

@dataclass(frozen=True)
class PrepInfo:
    orig_hw: Tuple[int, int]       # (H, W) original
    resized_hw: Tuple[int, int]    # (H, W) after resize (before pad)
    target_size: int               # 1008
    scale: float                   # target_size / max(orig_h, orig_w)


def preprocess_image_bgr(img_bgr: np.ndarray, target_size: int = 1008) -> Tuple[np.ndarray, PrepInfo]:
    """
    Returns:
      pixel_values: float32 [1,3,target_size,target_size]
      info: sizes + scale
    """
    H, W = img_bgr.shape[:2]
    scale = float(target_size) / float(max(H, W))
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # RGB -> float, rescale, normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    img_rgb = (img_rgb - _MEAN) / _STD

    # Pad to square (top-left anchored)
    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = img_rgb

    # NHWC -> NCHW
    pixel_values = np.transpose(padded, (2, 0, 1))[None, ...]

    info = PrepInfo(
        orig_hw=(H, W),
        resized_hw=(new_h, new_w),
        target_size=target_size,
        scale=scale,
    )
    return as_f32c(pixel_values), info


# ──────────────────────────────────────────────────────────────────────────────
# Prompt prep (original image coords -> resized/padded coords)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_points(points_xy: Iterable[Tuple[int, int]],
                   labels: Iterable[int],
                   info: PrepInfo) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(list(points_xy), dtype=np.float32)
    lbl = np.asarray(list(labels), dtype=np.int64)

    if pts.size == 0:
        pts = np.zeros((0, 2), np.float32)
        lbl = np.zeros((0,), np.int64)

    # scale from original to resized
    pts[:, 0] = pts[:, 0] * info.scale
    pts[:, 1] = pts[:, 1] * info.scale

    # shape to [1,1,N,2] and [1,1,N]
    pts = pts[None, None, :, :]
    lbl = lbl[None, None, :]

    return np.ascontiguousarray(pts), np.ascontiguousarray(lbl)


def prepare_box_as_points(rect_xyxy: Tuple[int, int, int, int],
                          info: PrepInfo) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = rect_xyxy
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    pts = [(x1, y1), (x2, y2)]
    lbl = [2, 3]  # top-left, bottom-right
    return prepare_points(pts, lbl, info)


# ──────────────────────────────────────────────────────────────────────────────
# Inference runners
# ──────────────────────────────────────────────────────────────────────────────

def run_encoder(sess_enc: InferenceSession, pixel_values: np.ndarray) -> Dict[str, np.ndarray]:
    inp0 = sess_enc.get_inputs()[0].name
    outs = sess_enc.run(None, {inp0: as_f32c(pixel_values)})
    out_names = [o.name for o in sess_enc.get_outputs()]
    return dict(zip(out_names, outs))


def _dtype_from_ort(ort_type: str):
    # Common ORT type strings: 'tensor(float)', 'tensor(float16)', 'tensor(int64)', ...
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


def run_decoder(sess_dec: InferenceSession,
                enc_out: Dict[str, np.ndarray],
                input_points: np.ndarray,
                input_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Feeds:
      - all decoder inputs that match encoder outputs by name
      - plus input_points / input_labels (name-matched by substring)
      - any remaining inputs get a conservative default (zeros)
    """
    feed: Dict[str, np.ndarray] = {}
    inputs = sess_dec.get_inputs()

    def _zeros(shape, dtype):
        # Replace dynamic dims (None) with 1
        shp = [(d if isinstance(d, int) and d >= 0 else 1) for d in shape]
        return np.zeros(shp, dtype=dtype)

    for inp in inputs:
        name = inp.name
        dtype = _dtype_from_ort(inp.type)

        if name in enc_out:
            arr = enc_out[name]
            # cast if needed
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            feed[name] = np.ascontiguousarray(arr)
            continue

        # Prompt inputs (common names in HF exports)
        lname = name.lower()
        if "input_points" in lname or "point_coords" in lname:
            feed[name] = np.ascontiguousarray(input_points.astype(dtype, copy=False))
            continue
        if "input_labels" in lname or "point_labels" in lname:
            feed[name] = np.ascontiguousarray(input_labels.astype(dtype, copy=False))
            continue

        # Optional mask-prompt inputs (set to "no mask")
        if "has_mask" in lname:
            z = _zeros(inp.shape, dtype)
            feed[name] = z
            continue
        if "mask_input" in lname or ("input_mask" in lname and "input_masks" in lname) or "input_masks" in lname:
            z = _zeros(inp.shape, dtype)
            feed[name] = z
            continue

        # Fallback zeros
        feed[name] = _zeros(inp.shape, dtype)

    outs = sess_dec.run(None, feed)
    out_names = [o.name for o in sess_dec.get_outputs()]
    return dict(zip(out_names, outs))


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing (approximate Sam*Processor.post_process_masks)
# ──────────────────────────────────────────────────────────────────────────────

def pick_best_mask(pred_masks: np.ndarray, iou_scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns: (mask_logits_2d, best_score)
    Handles common shapes:
      pred_masks: [B,Obj,M,H,W] or [B,M,H,W] or [M,H,W]
      iou_scores: [B,Obj,M] or [B,M] or [M]
    """
    m = pred_masks
    s = iou_scores

    # squeeze batch/object dims
    while m.ndim > 3:
        m = m[0]
    if m.ndim == 2:
        m = m[None, :, :]

    while s.ndim > 1:
        s = s[0]
    if s.ndim == 0:
        s = np.asarray([float(s)], dtype=np.float32)

    best = int(np.argmax(s))
    return m[best], float(s[best])


def postprocess_mask_to_original(mask_logits_2d: np.ndarray, info: PrepInfo) -> np.ndarray:
    """
    mask_logits_2d: [mask_h, mask_w] float
    returns: uint8 mask [H_orig, W_orig] with values 0 or 255
    """
    H0, W0 = info.orig_hw
    Hr, Wr = info.resized_hw
    T = info.target_size

    # 1) upsample to padded square
    up = cv2.resize(mask_logits_2d, (T, T), interpolation=cv2.INTER_LINEAR)

    # 2) crop padding (top-left)
    up = up[:Hr, :Wr]

    # 3) resize back to original
    up0 = cv2.resize(up, (W0, H0), interpolation=cv2.INTER_LINEAR)

    # threshold logits at 0
    mask = (up0 > 0.0).astype(np.uint8) * 255
    return mask


def green_overlay(bgr: np.ndarray, mask255: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    fg = mask255 > 0
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)


def compute_display_base(img_bgr: np.ndarray, max_side: int = 1200) -> Tuple[np.ndarray, float]:
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(W, H))
    disp = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    return disp, scale
