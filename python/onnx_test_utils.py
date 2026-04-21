# sam3-onnx-cpp/python/onnx_test_utils.py
"""
Utilities for SAM3-Tracker ONNX (image-only) testing.

Key behavior for this ONNX export:
- Encoder input: pixel_values [B, 3, 1008, 1008]
- Preprocess: direct resize to 1008x1008 (no padding), RGB, 1/255, mean/std=0.5
- Decoder inputs:
    input_points [B, 1, N, 2] float
    input_labels [B, 1, N]   int64
    input_boxes  [B, M, 4]   float
    image_embeddings.0/.1/.2

Important:
- For seed points mode, pass input_boxes as EMPTY [B,0,4] to avoid a dummy box prompt.
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

ACCEL = os.getenv("SAM3_ORT_ACCEL", "auto").lower()  # auto|cpu|cuda|trt

# Try preloading DLLs when CUDA EP exists (helps with pip-provided runtime DLLs)
try:
    if hasattr(ort, "preload_dlls") and "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()
except Exception:
    pass

_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
_STD  = np.array([0.5, 0.5, 0.5], np.float32)


def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] onnxruntime:", ort.__version__)
    print("[INFO] ORT providers (available):", ort.get_available_providers())
    print("[INFO] SAM3_ORT_ACCEL:", ACCEL)
    print("[INFO] SAM3_ORT_GRAPH_OPT:", os.getenv("SAM3_ORT_GRAPH_OPT", "all"))
    print("[INFO] SAM3_ORT_IO_BINDING:", os.getenv("SAM3_ORT_IO_BINDING", "auto"))


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


def _tensorrt_providers(model_path: str, device_id: int = 0):
    cache_root = Path(model_path).resolve().parent / "trt_cache" / Path(model_path).stem
    cache_root.mkdir(parents=True, exist_ok=True)
    use_fp16 = os.getenv("SAM3_TRT_FP16", "0").lower() not in ("0", "false", "no", "")
    return [
        ("TensorrtExecutionProvider", {
            "device_id": device_id,
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": str(cache_root),
            "trt_fp16_enable": "1" if use_fp16 else "0",
        }),
        *_cuda_providers(device_id=device_id),
    ]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return value


def _resolve_graph_optimization_level(safe: bool):
    if safe:
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    value = os.getenv("SAM3_ORT_GRAPH_OPT", "all").strip().lower()
    if value in ("disable", "disabled", "none", "off"):
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if value in ("basic",):
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if value in ("extended",):
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if value in ("all", "full", "aggressive"):
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    raise ValueError(
        "SAM3_ORT_GRAPH_OPT must be one of: disable, basic, extended, all."
    )


def make_session(path: str, tag: str = "model", safe: bool = False) -> InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = _resolve_graph_optimization_level(safe)
    so.intra_op_num_threads = _env_int(
        "SAM3_ORT_INTRA_OP_THREADS",
        max(1, (os.cpu_count() or 8) - 1),
    )
    so.inter_op_num_threads = _env_int("SAM3_ORT_INTER_OP_THREADS", 1)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = os.getenv("SAM3_ORT_CPU_ARENA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )
    so.enable_mem_pattern = os.getenv("SAM3_ORT_MEM_PATTERN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )

    path = str(Path(path).resolve())
    av = ort.get_available_providers()
    if ACCEL == "cpu":
        providers = ["CPUExecutionProvider"]
    elif ACCEL == "trt":
        if "TensorrtExecutionProvider" in av:
            providers = _tensorrt_providers(path)
        elif "CUDAExecutionProvider" in av:
            providers = _cuda_providers()
        else:
            providers = ["CPUExecutionProvider"]
    elif ACCEL == "cuda":
        providers = _cuda_providers() if "CUDAExecutionProvider" in av else ["CPUExecutionProvider"]
    else:  # auto
        providers = _cuda_providers() if "CUDAExecutionProvider" in av else ["CPUExecutionProvider"]

    print(f"[INFO] Loading {os.path.basename(path)} [{tag}] providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=list(providers))
    if (
        ACCEL == "trt"
        and sess.get_providers() == ["CPUExecutionProvider"]
        and "CUDAExecutionProvider" in av
    ):
        print(
            "[WARN] TensorRT EP was requested but is not usable in this environment; "
            "retrying with CUDAExecutionProvider."
        )
        sess = InferenceSession(path, sess_options=so, providers=list(_cuda_providers()))
    print("[INFO] Active providers:", sess.get_providers())
    return sess


def as_f32c(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


@dataclass(frozen=True)
class PrepInfo:
    orig_hw: Tuple[int, int]   # (H, W)
    target_size: int           # 1008
    scale_x: float             # target_size / W
    scale_y: float             # target_size / H


def preprocess_image_bgr(img_bgr: np.ndarray, target_size: int = 1008) -> Tuple[np.ndarray, PrepInfo]:
    H, W = img_bgr.shape[:2]
    scale_x = float(target_size) / float(W)
    scale_y = float(target_size) / float(H)

    img_resized = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    img_rgb = (img_rgb - _MEAN) / _STD

    pixel_values = np.transpose(img_rgb, (2, 0, 1))[None, ...]
    info = PrepInfo(orig_hw=(H, W), target_size=target_size, scale_x=scale_x, scale_y=scale_y)
    return as_f32c(pixel_values), info


def empty_points() -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros((1, 1, 0, 2), np.float32), np.zeros((1, 1, 0), np.int64)


def empty_boxes() -> np.ndarray:
    return np.zeros((1, 0, 4), np.float32)


def prepare_points(points_xy: Iterable[Tuple[int, int]], labels: Iterable[int], info: PrepInfo):
    pts = np.asarray(list(points_xy), dtype=np.float32)
    lbl = np.asarray(list(labels), dtype=np.int64)
    if pts.size == 0:
        return empty_points()

    pts[:, 0] *= info.scale_x
    pts[:, 1] *= info.scale_y

    pts = pts[None, None, :, :]
    lbl = lbl[None, None, :]
    return np.ascontiguousarray(pts), np.ascontiguousarray(lbl)


def prepare_boxes(rect_xyxy: Tuple[int, int, int, int], info: PrepInfo) -> np.ndarray:
    x1, y1, x2, y2 = rect_xyxy
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = np.array([[x1 * info.scale_x, y1 * info.scale_y, x2 * info.scale_x, y2 * info.scale_y]], np.float32)
    return np.ascontiguousarray(box[None, :, :])  # [1,1,4]


def run_encoder(sess_enc: InferenceSession, pixel_values: np.ndarray) -> Dict[str, np.ndarray]:
    inp = sess_enc.get_inputs()[0].name
    outs = sess_enc.run(None, {inp: as_f32c(pixel_values)})
    names = [o.name for o in sess_enc.get_outputs()]
    return dict(zip(names, outs))


def run_decoder(sess_dec: InferenceSession,
                enc_out: Dict[str, np.ndarray],
                input_points: Optional[np.ndarray] = None,
                input_labels: Optional[np.ndarray] = None,
                input_boxes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    if input_points is None or input_labels is None:
        input_points, input_labels = empty_points()
    if input_boxes is None:
        input_boxes = empty_boxes()

    feed = {
        "input_points": np.ascontiguousarray(input_points.astype(np.float32, copy=False)),
        "input_labels": np.ascontiguousarray(input_labels.astype(np.int64, copy=False)),
        "input_boxes":  np.ascontiguousarray(input_boxes.astype(np.float32, copy=False)),
        "image_embeddings.0": np.ascontiguousarray(enc_out["image_embeddings.0"]),
        "image_embeddings.1": np.ascontiguousarray(enc_out["image_embeddings.1"]),
        "image_embeddings.2": np.ascontiguousarray(enc_out["image_embeddings.2"]),
    }

    outs = sess_dec.run(None, feed)
    names = [o.name for o in sess_dec.get_outputs()]
    return dict(zip(names, outs))


def pick_best_mask(pred_masks: np.ndarray, iou_scores: np.ndarray, which_prompt: int = 0):
    m = pred_masks[0, which_prompt]   # [3,H,W]
    s = iou_scores[0, which_prompt]   # [3]
    best = int(np.argmax(s))
    return m[best], float(s[best])


def postprocess_mask_to_original(mask_logits_2d: np.ndarray, info: PrepInfo) -> np.ndarray:
    H0, W0 = info.orig_hw
    T = info.target_size
    up = cv2.resize(mask_logits_2d, (T, T), interpolation=cv2.INTER_LINEAR)
    up0 = cv2.resize(up, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return (up0 > 0.0).astype(np.uint8) * 255


def green_overlay(bgr: np.ndarray, mask255: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    fg = (mask255 > 0)
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)


def compute_display_base(img_bgr: np.ndarray, max_side: int = 1200):
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(W, H))
    disp = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    return disp, scale
