#!/usr/bin/env python3
from __future__ import annotations

import json
import copy
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from onnx_runtime_policy import (
    DEFAULT_MAX_MEM_FRAMES,
    DEFAULT_MAX_OBJ_PTRS,
    MULTI_GRAPH_PROFILE,
    SINGLE_GRAPH_PROFILE,
)
from onnx_test_utils import PrepInfo, green_overlay, make_session, preprocess_image_bgr


DEFAULT_NUM_MASKMEM = 7
DEFAULT_MAX_COND_FRAMES_IN_ATTN = 4
DEFAULT_MEMORY_TEMPORAL_STRIDE_FOR_EVAL = 1

ORT_TENSOR_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def _as_f32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _as_i32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.int32, copy=False))


def _to_numpy(value: np.ndarray | ort.OrtValue) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.ascontiguousarray(value.numpy())


def _sigmoid_np(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-value))


def _fixed_meta_shape(shape: list[Any] | tuple[Any, ...]) -> tuple[int, ...] | None:
    dims: list[int] = []
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        dims.append(int(dim))
    return tuple(dims)


def _select_closest_cond_frames(
    frame_idx: int,
    cond_frame_outputs: dict[int, Any],
    max_cond_frame_num: int,
    *,
    keep_first_cond_frame: bool = False,
) -> tuple[dict[int, Any], dict[int, Any]]:
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}

    if max_cond_frame_num < 2:
        raise ValueError("max_cond_frame_num must be -1 or >= 2")

    selected_outputs: dict[int, Any] = {}
    if keep_first_cond_frame:
        idx_first = min((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_first is None:
            idx_first = max((t for t in cond_frame_outputs if t > frame_idx), default=None)
        if idx_first is not None:
            selected_outputs[idx_first] = cond_frame_outputs[idx_first]

    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected_outputs[idx_before] = cond_frame_outputs[idx_before]

    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected_outputs[idx_after] = cond_frame_outputs[idx_after]

    num_remain = max_cond_frame_num - len(selected_outputs)
    inds_remain = sorted(
        (t for t in cond_frame_outputs if t not in selected_outputs),
        key=lambda value: abs(value - frame_idx),
    )[:num_remain]
    selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
    unselected_outputs = {
        t: value for t, value in cond_frame_outputs.items() if t not in selected_outputs
    }
    return selected_outputs, unselected_outputs


def empty_prompt() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((1, 0, 2), np.float32), np.zeros((1, 0), np.int32)


def prepare_prompt_points(
    points_xy: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    labels: list[int] | tuple[int, ...],
    info: PrepInfo,
) -> tuple[np.ndarray, np.ndarray]:
    if not points_xy:
        return empty_prompt()
    points = np.asarray(points_xy, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)
    points[:, 0] *= info.scale_x
    points[:, 1] *= info.scale_y
    return np.ascontiguousarray(points[None, ...]), np.ascontiguousarray(point_labels[None, ...])


def prepare_prompt_box(
    rect_xyxy: tuple[int, int, int, int] | None,
    info: PrepInfo,
) -> tuple[np.ndarray, np.ndarray]:
    if rect_xyxy is None:
        return empty_prompt()
    x1, y1, x2, y2 = rect_xyxy
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    points[:, 0] *= info.scale_x
    points[:, 1] *= info.scale_y
    labels = np.array([2, 3], dtype=np.int32)
    return np.ascontiguousarray(points[None, ...]), np.ascontiguousarray(labels[None, ...])


def parse_points_text(text: str) -> tuple[list[tuple[int, int]], list[int]]:
    points, labels = [], []
    if not text.strip():
        return points, labels
    for item in text.split(";"):
        x_str, y_str, label_str = [part.strip() for part in item.split(",")]
        points.append((int(float(x_str)), int(float(y_str))))
        labels.append(int(label_str))
    return points, labels


def parse_box_text(text: str) -> tuple[int, int, int, int]:
    parts = [int(float(part.strip())) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--box expects x1,y1,x2,y2")
    return tuple(parts)


def _binary_mask(mask_hw: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_hw)
    if mask.ndim != 2:
        raise ValueError(f"Mask prompts expect a 2D array, got shape {mask.shape}.")
    if np.issubdtype(mask.dtype, np.floating):
        mask_bin = mask > 0.5
    else:
        mask_bin = mask > 0
    return np.ascontiguousarray(mask_bin.astype(np.uint8, copy=False))


def _mask_bbox_xyxy(mask_hw: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask_hw)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_prompt_from_box(box_xyxy: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = box_xyxy
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    labels = np.array([2, 3], dtype=np.int32)
    return np.ascontiguousarray(points[None, ...]), np.ascontiguousarray(labels[None, ...])


def _mask_prompt_from_point(point_xy: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    point = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    return np.ascontiguousarray(point[None, ...]), np.ascontiguousarray(labels[None, ...])


def load_prompt_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if "prompt" not in spec:
        raise SystemExit(f"Prompt JSON is missing 'prompt': {path}")
    return spec


def save_prompt_spec(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


def _mask_logits_candidates(mask_logits_high_res: np.ndarray) -> np.ndarray:
    mask_logits = np.asarray(mask_logits_high_res)
    if mask_logits.ndim == 2:
        return np.ascontiguousarray(mask_logits[None, ...])
    if mask_logits.ndim == 3:
        return np.ascontiguousarray(mask_logits)
    if mask_logits.ndim == 4:
        if mask_logits.shape[0] == 1:
            return np.ascontiguousarray(mask_logits[0])
        if mask_logits.shape[1] == 1:
            return np.ascontiguousarray(mask_logits[:, 0])
        return np.ascontiguousarray(mask_logits.reshape((-1, mask_logits.shape[-2], mask_logits.shape[-1])))
    if mask_logits.ndim == 5:
        return np.ascontiguousarray(mask_logits.reshape((-1, mask_logits.shape[-2], mask_logits.shape[-1])))
    raise ValueError(f"Expected mask logits with 2-5 dimensions, got shape {mask_logits.shape}.")


def masks_to_uint8(mask_logits_high_res: np.ndarray, info: PrepInfo) -> np.ndarray:
    masks = []
    for mask_logits in _mask_logits_candidates(mask_logits_high_res):
        mask_resized = cv2.resize(
            mask_logits,
            (info.orig_hw[1], info.orig_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        masks.append((mask_resized > 0.0).astype(np.uint8) * 255)
    if not masks:
        return np.zeros((0, info.orig_hw[0], info.orig_hw[1]), dtype=np.uint8)
    return np.stack(masks).astype(np.uint8, copy=False)


def mask_to_uint8(mask_logits_high_res: np.ndarray, info: PrepInfo) -> np.ndarray:
    masks = masks_to_uint8(mask_logits_high_res, info)
    if masks.shape[0] == 0:
        return np.zeros(info.orig_hw, dtype=np.uint8)
    return masks[0]


def mask_to_overlay(frame_bgr: np.ndarray, mask_logits_high_res: np.ndarray, info: PrepInfo) -> np.ndarray:
    mask_uint8 = mask_to_uint8(mask_logits_high_res, info)
    return green_overlay(frame_bgr, mask_uint8, alpha=0.5)


def prepare_prompt_mask(
    mask_hw: np.ndarray,
    info: PrepInfo,
    *,
    prompt_strategy: str = "box",
) -> "PreparedMaskPrompt":
    if prompt_strategy not in {"box", "point"}:
        raise ValueError("prompt_strategy must be 'box' or 'point'.")

    mask_bin = _binary_mask(mask_hw)
    target_hw = (int(info.target_size), int(info.target_size))
    if mask_bin.shape == info.orig_hw:
        mask_target_f32 = cv2.resize(
            mask_bin.astype(np.float32, copy=False),
            (info.target_size, info.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_target = np.ascontiguousarray((mask_target_f32 > 0.5).astype(np.uint8, copy=False))
        mask_uint8 = np.ascontiguousarray(mask_bin.astype(np.uint8, copy=False) * 255)
    elif mask_bin.shape == target_hw:
        mask_target = mask_bin
        mask_uint8 = None
    else:
        raise ValueError(
            "Mask prompts must match either the original frame size "
            f"{info.orig_hw} or the resized ONNX size {target_hw}; got {mask_bin.shape}."
        )

    bbox = _mask_bbox_xyxy(mask_target)
    if bbox is None:
        raise ValueError("Mask prompt cannot be empty.")

    x1, y1, x2, y2 = bbox
    use_point_prompt = prompt_strategy == "point" or x1 == x2 or y1 == y2
    if use_point_prompt:
        point = ((x1 + x2) // 2, (y1 + y2) // 2)
        prompt_points, prompt_labels = _mask_prompt_from_point(point)
        prompt_kind = "seed_points"
    else:
        prompt_points, prompt_labels = _mask_prompt_from_box(bbox)
        prompt_kind = "bounding_box"

    mask_logits_high_res = np.where(mask_target > 0, 20.0, -20.0).astype(np.float32, copy=False)
    mask_logits_high_res = np.ascontiguousarray(mask_logits_high_res[None, None, ...])
    if mask_uint8 is None:
        mask_uint8 = mask_to_uint8(mask_logits_high_res, info)

    return PreparedMaskPrompt(
        mask_logits_high_res=mask_logits_high_res,
        mask_uint8=mask_uint8,
        prompt_points=prompt_points,
        prompt_labels=prompt_labels,
        prompt_kind=prompt_kind,
    )


def normalize_encoder_outputs(
    raw_outputs: dict[str, np.ndarray | ort.OrtValue],
    constants: dict[str, np.ndarray],
) -> dict[str, np.ndarray | ort.OrtValue]:
    if "image_embeddings" in raw_outputs:
        return raw_outputs

    current_vision_feat = raw_outputs["image_embeddings.2"]
    return {
        "image_embeddings": _as_f32c(_to_numpy(current_vision_feat) + constants["no_mem_embed_bchw"]),
        "high_res_features0": raw_outputs["image_embeddings.0"],
        "high_res_features1": raw_outputs["image_embeddings.1"],
        "current_vision_feat": current_vision_feat,
        "current_vision_pos_embed": constants["current_vision_pos_embed"],
    }


@dataclass(frozen=True)
class PreparedFrame:
    encoder_outputs: dict[str, np.ndarray | ort.OrtValue]
    info: PrepInfo
    prep_ms: float
    enc_ms: float


@dataclass(frozen=True)
class PreparedMaskPrompt:
    mask_logits_high_res: np.ndarray
    mask_uint8: np.ndarray
    prompt_points: np.ndarray
    prompt_labels: np.ndarray
    prompt_kind: str


@dataclass(frozen=True)
class PromptMaskCandidates:
    selected_mask_logits_high_res: np.ndarray
    selected_mask_uint8: np.ndarray
    multimask_logits_high_res: np.ndarray | None = None
    multimask_uint8: np.ndarray | None = None
    iou_scores: np.ndarray | None = None


@dataclass(frozen=True)
class TrackerFrameState:
    maskmem_features: np.ndarray | ort.OrtValue
    maskmem_pos_enc: np.ndarray | ort.OrtValue
    obj_ptr: np.ndarray | ort.OrtValue
    object_score_logits: np.ndarray | ort.OrtValue | None = None
    eff_iou_score: float | None = None


@dataclass(frozen=True)
class TrackerMemorySnapshot:
    cond_states: dict[int, TrackerFrameState]
    non_cond_states: dict[int, TrackerFrameState]


@dataclass(frozen=True)
class FrameTimings:
    prep_ms: float
    enc_ms: float
    attn_ms: float
    dec_ms: float
    mem_ms: float
    total_ms: float


@dataclass(frozen=True)
class FrameResult:
    frame_idx: int
    info: PrepInfo
    state: TrackerFrameState
    mask_uint8: np.ndarray
    timings: FrameTimings


@dataclass(frozen=True)
class _ResolvedTrackerBundle:
    graph_profile: str
    precision: str
    constants_path: Path
    decoder_path: Path
    memory_attention_path: Path
    memory_encoder_path: Path


@dataclass
class _InputBuffer:
    array: np.ndarray
    ortvalue: ort.OrtValue | None

    def sync(self) -> np.ndarray | ort.OrtValue:
        if self.ortvalue is not None:
            self.ortvalue.update_inplace(self.array)
            return self.ortvalue
        return self.array


class _OrtSessionRunner:
    def __init__(self, session, tag: str) -> None:
        self.session = session
        self.tag = tag
        providers = session.get_providers()
        self.device_type = (
            "cuda"
            if any(
                provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
                for provider in providers
            )
            else "cpu"
        )
        io_binding_pref = os.getenv("SAM3_ORT_IO_BINDING", "auto").strip().lower()
        self.enable_iobinding = io_binding_pref == "on" or (
            io_binding_pref == "auto" and self.device_type != "cpu"
        )
        self._output_buffers: dict[str, ort.OrtValue] = {}
        self._static_output_shapes = {
            output.name: _fixed_meta_shape(output.shape)
            for output in self.session.get_outputs()
        }

    def _bind_input(self, binding, name: str, value, owned_inputs: list[ort.OrtValue]) -> None:
        if isinstance(value, ort.OrtValue):
            binding.bind_ortvalue_input(name, value)
            return

        arr = np.ascontiguousarray(value)
        if self.device_type == "cpu":
            binding.bind_cpu_input(name, arr)
            return

        ort_value = ort.OrtValue.ortvalue_from_numpy(arr, self.device_type, 0)
        owned_inputs.append(ort_value)
        binding.bind_ortvalue_input(name, ort_value)

    def run(
        self,
        feeds: dict[str, np.ndarray | ort.OrtValue],
        output_device: str = "cpu",
    ) -> dict[str, np.ndarray | ort.OrtValue]:
        output_names = [output.name for output in self.session.get_outputs()]
        must_use_iobinding = self.enable_iobinding or any(
            isinstance(value, ort.OrtValue) for value in feeds.values()
        )

        if not must_use_iobinding and output_device == "cpu":
            values = self.session.run(None, feeds)
            return dict(zip(output_names, values))

        binding = self.session.io_binding()
        owned_inputs: list[ort.OrtValue] = []
        for name, value in feeds.items():
            self._bind_input(binding, name, value, owned_inputs)

        bind_device = (
            self.device_type if output_device == "session" and self.device_type != "cpu" else "cpu"
        )
        bound_outputs: dict[str, ort.OrtValue] = {}
        for output_name in output_names:
            if bind_device != "cpu":
                output_meta = next(
                    output for output in self.session.get_outputs() if output.name == output_name
                )
                shape = self._static_output_shapes.get(output_name)
                dtype = ORT_TENSOR_TYPE_TO_NUMPY.get(output_meta.type)
                if shape is not None and dtype is not None:
                    buffer = self._output_buffers.get(output_name)
                    if buffer is None:
                        buffer = ort.OrtValue.ortvalue_from_shape_and_type(
                            shape,
                            dtype,
                            bind_device,
                            0,
                        )
                        self._output_buffers[output_name] = buffer
                    binding.bind_ortvalue_output(output_name, buffer)
                    bound_outputs[output_name] = buffer
                    continue
            binding.bind_output(output_name, bind_device, 0)

        self.session.run_with_iobinding(binding)
        if bind_device == "cpu":
            values = binding.copy_outputs_to_cpu()
            return dict(zip(output_names, values))

        binding.synchronize_outputs()
        if len(bound_outputs) == len(output_names):
            return {name: bound_outputs[name] for name in output_names}
        values = list(binding.get_outputs())
        return dict(zip(output_names, values))


class Sam3OnnxTrackerSession:
    def __init__(
        self,
        onnx_dir: Path,
        *,
        safe: bool = False,
        max_mem_frames: int = DEFAULT_MAX_MEM_FRAMES,
        max_obj_ptrs: int = DEFAULT_MAX_OBJ_PTRS,
    ) -> None:
        self.onnx_dir = Path(onnx_dir).resolve()
        self.mode = "default"
        requested_graph_profile = self._resolve_graph_profile(int(max_mem_frames))
        tracker_bundle = self._resolve_tracker_bundle(self.onnx_dir, requested_graph_profile)
        self.graph_profile = tracker_bundle.graph_profile
        self.tracker_precision = tracker_bundle.precision
        self.constants_path = tracker_bundle.constants_path
        self.constants = self._load_video_constants(self.constants_path)

        native_num_maskmem = int(
            self.constants.get("num_maskmem", np.array([DEFAULT_NUM_MASKMEM], dtype=np.int64))[0]
        )
        native_max_obj_ptrs = int(
            self.constants.get("max_obj_ptrs", np.array([DEFAULT_MAX_OBJ_PTRS], dtype=np.int64))[0]
        )
        requested_num_maskmem = (
            max(1, min(native_num_maskmem, int(max_mem_frames)))
            if max_mem_frames > 0
            else native_num_maskmem
        )
        requested_max_obj_ptrs = (
            max(1, min(native_max_obj_ptrs, int(max_obj_ptrs)))
            if max_obj_ptrs > 0
            else native_max_obj_ptrs
        )

        enc_path = self._resolve_encoder_path(self.onnx_dir)
        dec_path = tracker_bundle.decoder_path
        mat_path = tracker_bundle.memory_attention_path
        men_path = tracker_bundle.memory_encoder_path

        for path in (enc_path, dec_path, mat_path, men_path):
            if not path.exists():
                raise SystemExit(f"Missing ONNX file: {path}")

        self.model_paths = {
            "encoder": enc_path,
            "decoder": dec_path,
            "memory_attention": mat_path,
            "memory_encoder": men_path,
        }
        self.sess_enc = make_session(str(enc_path), tag="video_image_encoder", safe=safe)
        self.sess_dec = make_session(str(dec_path), tag="video_image_decoder", safe=safe)
        self.sess_mat = make_session(str(mat_path), tag="video_memory_attention", safe=safe)
        self.sess_men = make_session(str(men_path), tag="video_memory_encoder", safe=safe)
        self.decoder_output_names = {output.name for output in self.sess_dec.get_outputs()}
        self.decoder_has_iou_scores = "iou_scores" in self.decoder_output_names
        self.decoder_has_multimasks = "pred_multimasks_high_res" in self.decoder_output_names

        self._enc_runner = _OrtSessionRunner(self.sess_enc, "video_image_encoder")
        self._dec_runner = _OrtSessionRunner(self.sess_dec, "video_image_decoder")
        self._mat_runner = _OrtSessionRunner(self.sess_mat, "video_memory_attention")
        self._men_runner = _OrtSessionRunner(self.sess_men, "video_memory_encoder")
        self.device_type = self._mat_runner.device_type

        self.static_num_mem_frames = self._fixed_input_dim(self.sess_mat, "memory_mask_feats", 0)
        self.static_num_obj_ptrs = self._fixed_input_dim(self.sess_mat, "memory_obj_ptrs", 0)
        self.num_maskmem = (
            min(int(self.static_num_mem_frames), requested_num_maskmem)
            if self.static_num_mem_frames is not None
            else requested_num_maskmem
        )
        self.max_obj_ptrs = (
            min(int(self.static_num_obj_ptrs), requested_max_obj_ptrs)
            if self.static_num_obj_ptrs is not None
            else requested_max_obj_ptrs
        )
        self.max_cond_frames_in_attn = int(
            self.constants.get(
                "max_cond_frames_in_attn",
                np.array([DEFAULT_MAX_COND_FRAMES_IN_ATTN], dtype=np.int64),
            )[0]
        )
        self.keep_first_cond_frame = bool(
            int(self.constants.get("keep_first_cond_frame", np.array([0], dtype=np.int64))[0])
        )
        self.memory_temporal_stride_for_eval = int(
            self.constants.get(
                "memory_temporal_stride_for_eval",
                np.array([DEFAULT_MEMORY_TEMPORAL_STRIDE_FOR_EVAL], dtype=np.int64),
            )[0]
        )
        self.use_memory_selection = bool(
            int(self.constants.get("use_memory_selection", np.array([0], dtype=np.int64))[0])
        )
        self.mf_threshold = float(
            self.constants.get("mf_threshold", np.array([0.01], dtype=np.float32))[0]
        )
        self._constant_ortvalues: dict[str, ort.OrtValue] = {}
        if self.device_type != "cpu" and "current_vision_pos_embed" in self.constants:
            self._constant_ortvalues["current_vision_pos_embed"] = ort.OrtValue.ortvalue_from_numpy(
                np.ascontiguousarray(self.constants["current_vision_pos_embed"]),
                self.device_type,
                0,
            )
        self._memory_input_buffers = self._build_memory_input_buffers()
        self.warmup_enabled = self._resolve_warmup_enabled()

        self.reset()
        if self.warmup_enabled:
            self._warmup_runtime()

    @property
    def uses_iobinding(self) -> bool:
        return (
            self._enc_runner.enable_iobinding
            or self._dec_runner.enable_iobinding
            or self._mat_runner.enable_iobinding
            or self._men_runner.enable_iobinding
        )

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "graph_profile": self.graph_profile,
            "tracker_precision": self.tracker_precision,
            "model_names": {name: path.name for name, path in self.model_paths.items()},
            "model_paths": {name: str(path) for name, path in self.model_paths.items()},
            "constants_path": str(self.constants_path),
            "providers": {
                "encoder": self.sess_enc.get_providers(),
                "decoder": self.sess_dec.get_providers(),
                "memory_attention": self.sess_mat.get_providers(),
                "memory_encoder": self.sess_men.get_providers(),
            },
            "device_type": self.device_type,
            "uses_iobinding": self.uses_iobinding,
            "warmup_enabled": self.warmup_enabled,
            "static_num_mem_frames": self.static_num_mem_frames,
            "static_num_obj_ptrs": self.static_num_obj_ptrs,
            "effective_max_mem_frames": self.num_maskmem,
            "effective_max_obj_ptrs": self.max_obj_ptrs,
            "max_cond_frames_in_attn": self.max_cond_frames_in_attn,
            "keep_first_cond_frame": self.keep_first_cond_frame,
            "memory_temporal_stride_for_eval": self.memory_temporal_stride_for_eval,
            "use_memory_selection": self.use_memory_selection,
            "mf_threshold": self.mf_threshold,
            "decoder_has_iou_scores": self.decoder_has_iou_scores,
            "decoder_has_multimasks": self.decoder_has_multimasks,
            "max_non_cond_frames_kept": self._max_non_cond_history(),
        }

    def reset(self) -> None:
        self.cond_states: dict[int, TrackerFrameState] = {}
        self.non_cond_states: dict[int, TrackerFrameState] = {}

    def capture_memory_snapshot(self) -> TrackerMemorySnapshot:
        return TrackerMemorySnapshot(
            cond_states=copy.deepcopy(self.cond_states),
            non_cond_states=copy.deepcopy(self.non_cond_states),
        )

    def restore_memory_snapshot(self, snapshot: TrackerMemorySnapshot) -> None:
        self.cond_states = copy.deepcopy(snapshot.cond_states)
        self.non_cond_states = copy.deepcopy(snapshot.non_cond_states)

    def capture_state_snapshot(self) -> TrackerMemorySnapshot:
        return self.capture_memory_snapshot()

    def restore_state_snapshot(self, snapshot: TrackerMemorySnapshot) -> None:
        self.restore_memory_snapshot(snapshot)

    def prepare_frame(self, frame_bgr: np.ndarray) -> PreparedFrame:
        t_prep = time.time()
        pixel_values, info = preprocess_image_bgr(frame_bgr, target_size=1008)
        prep_ms = (time.time() - t_prep) * 1000.0

        input_name = self.sess_enc.get_inputs()[0].name
        output_device = "session" if self._enc_runner.device_type != "cpu" else "cpu"
        t_enc = time.time()
        raw_outputs = self._enc_runner.run({input_name: _as_f32c(pixel_values)}, output_device=output_device)
        enc_ms = (time.time() - t_enc) * 1000.0
        encoder_outputs = normalize_encoder_outputs(raw_outputs, self.constants)
        if "current_vision_pos_embed" in self._constant_ortvalues:
            encoder_outputs["current_vision_pos_embed"] = self._constant_ortvalues[
                "current_vision_pos_embed"
            ]
        return PreparedFrame(
            encoder_outputs=encoder_outputs,
            info=info,
            prep_ms=prep_ms,
            enc_ms=enc_ms,
        )

    def prepare_prompt_from_spec(
        self,
        frame_bgr: np.ndarray,
        prompt_spec: dict[str, Any],
    ) -> tuple[PreparedFrame, np.ndarray, np.ndarray]:
        prepared = self.prepare_frame(frame_bgr)
        prompt_kind = prompt_spec["prompt"]
        if prompt_kind == "bounding_box":
            box = prompt_spec.get("box")
            prompt_points, prompt_labels = prepare_prompt_box(
                tuple(box) if box is not None else None,
                prepared.info,
            )
        elif prompt_kind == "seed_points":
            raw_points = prompt_spec.get("points", [])
            points = [(int(item[0]), int(item[1])) for item in raw_points]
            labels = [int(item[2]) for item in raw_points]
            prompt_points, prompt_labels = prepare_prompt_points(points, labels, prepared.info)
        else:
            raise SystemExit(f"Unsupported prompt kind in prompt JSON: {prompt_kind}")
        return prepared, prompt_points, prompt_labels

    def preview_prompt_mask(
        self,
        prepared: PreparedFrame,
        prompt_points: np.ndarray,
        prompt_labels: np.ndarray,
    ) -> np.ndarray:
        return self.preview_prompt_candidates(
            prepared,
            prompt_points,
            prompt_labels,
        ).selected_mask_logits_high_res

    def preview_prompt_candidates(
        self,
        prepared: PreparedFrame,
        prompt_points: np.ndarray,
        prompt_labels: np.ndarray,
    ) -> PromptMaskCandidates:
        dec = self._run_decoder(
            prompt_points,
            prompt_labels,
            prepared.encoder_outputs["image_embeddings"],
            prepared.encoder_outputs["high_res_features0"],
            prepared.encoder_outputs["high_res_features1"],
            output_device="cpu",
        )
        selected_logits = _to_numpy(dec["pred_mask_high_res"])
        selected_uint8 = mask_to_uint8(selected_logits, prepared.info)
        multimask_logits = None
        multimask_uint8 = None
        if "pred_multimasks_high_res" in dec:
            multimask_logits = _to_numpy(dec["pred_multimasks_high_res"])
            multimask_uint8 = masks_to_uint8(multimask_logits, prepared.info)
        iou_scores = _to_numpy(dec["iou_scores"]) if "iou_scores" in dec else None
        return PromptMaskCandidates(
            selected_mask_logits_high_res=selected_logits,
            selected_mask_uint8=selected_uint8,
            multimask_logits_high_res=multimask_logits,
            multimask_uint8=multimask_uint8,
            iou_scores=iou_scores,
        )

    def _condition_with_mask_prompt(
        self,
        prepared: PreparedFrame,
        prompt_mask: np.ndarray,
        *,
        prompt_points: np.ndarray | None,
        prompt_labels: np.ndarray | None,
        prompt_mask_strategy: str,
    ) -> tuple[TrackerFrameState, np.ndarray, float, float]:
        enc = prepared.encoder_outputs
        prepared_mask = prepare_prompt_mask(
            prompt_mask,
            prepared.info,
            prompt_strategy=prompt_mask_strategy,
        )

        has_explicit_prompt = (
            prompt_points is not None
            and prompt_labels is not None
            and np.asarray(prompt_labels).size > 0
        )
        if not has_explicit_prompt:
            prompt_points = prepared_mask.prompt_points
            prompt_labels = prepared_mask.prompt_labels

        t_dec = time.time()
        dec = self._run_decoder(
            prompt_points,
            prompt_labels,
            enc["image_embeddings"],
            enc["high_res_features0"],
            enc["high_res_features1"],
            output_device="session",
        )
        dec_ms = (time.time() - t_dec) * 1000.0

        # The object is explicitly present in a user-provided mask, so force a positive
        # object-presence signal even though the exported decoder cannot ingest masks directly.
        object_present_logits = np.ascontiguousarray(np.full((1, 1), 10.0, dtype=np.float32))

        t_mem = time.time()
        mem = self._run_memory_encoder(
            prepared_mask.mask_logits_high_res,
            enc["current_vision_feat"],
            object_present_logits,
            is_mask_from_points=True,
            output_device="session",
        )
        mem_ms = (time.time() - t_mem) * 1000.0

        state = TrackerFrameState(
            maskmem_features=_to_numpy(mem["maskmem_features"]),
            maskmem_pos_enc=_to_numpy(mem["maskmem_pos_enc"]),
            obj_ptr=_to_numpy(dec["obj_ptr"]),
            object_score_logits=object_present_logits,
            eff_iou_score=1.0,
        )
        return state, prepared_mask.mask_uint8, dec_ms, mem_ms

    def process_frame(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        *,
        prepared: PreparedFrame | None = None,
        prompt_points: np.ndarray | None = None,
        prompt_labels: np.ndarray | None = None,
        prompt_mask: np.ndarray | None = None,
        prompt_mask_strategy: str = "box",
    ) -> FrameResult:
        t_total = time.time()
        has_point_prompt = (
            prompt_points is not None
            and prompt_labels is not None
            and np.asarray(prompt_labels).size > 0
        )
        has_mask_prompt = prompt_mask is not None and np.asarray(prompt_mask).size > 0
        has_prompt = has_point_prompt or has_mask_prompt
        is_conditioning_frame = frame_idx == 0 or has_prompt

        if is_conditioning_frame and has_mask_prompt:
            if prepared is None:
                prepared = self.prepare_frame(frame_bgr)
            info = prepared.info
            prep_ms = prepared.prep_ms
            enc_ms = prepared.enc_ms

            state, mask_uint8, dec_ms, mem_ms = self._condition_with_mask_prompt(
                prepared,
                prompt_mask,
                prompt_points=prompt_points,
                prompt_labels=prompt_labels,
                prompt_mask_strategy=prompt_mask_strategy,
            )
            self.non_cond_states.pop(frame_idx, None)
            self.cond_states[frame_idx] = state
            return FrameResult(
                frame_idx=frame_idx,
                info=info,
                state=state,
                mask_uint8=mask_uint8,
                timings=FrameTimings(
                    prep_ms=prep_ms,
                    enc_ms=enc_ms,
                    attn_ms=0.0,
                    dec_ms=dec_ms,
                    mem_ms=mem_ms,
                    total_ms=(time.time() - t_total) * 1000.0,
                ),
            )

        if is_conditioning_frame:
            if prepared is None:
                prepared = self.prepare_frame(frame_bgr)
            enc = prepared.encoder_outputs
            info = prepared.info
            prep_ms = prepared.prep_ms
            enc_ms = prepared.enc_ms
            fused_embed = enc["image_embeddings"]
            prompt_points = prompt_points if prompt_points is not None else empty_prompt()[0]
            prompt_labels = prompt_labels if prompt_labels is not None else empty_prompt()[1]
            is_mask_from_points = True
            attn_ms = 0.0
        else:
            if prepared is None:
                prepared = self.prepare_frame(frame_bgr)
            enc = prepared.encoder_outputs
            info = prepared.info
            prep_ms = prepared.prep_ms
            enc_ms = prepared.enc_ms

            mem_inputs = self._select_memory_inputs(frame_idx)
            t_attn = time.time()
            fused_embed = self._run_memory_attention(
                enc["current_vision_feat"],
                enc["current_vision_pos_embed"],
                mem_inputs["memory_obj_ptrs"],
                mem_inputs["memory_obj_tpos"],
                mem_inputs["memory_mask_feats"],
                mem_inputs["memory_mask_pos"],
                mem_inputs["memory_mask_tpos_idx"],
                output_device="session",
            )["fused_feat"]
            attn_ms = (time.time() - t_attn) * 1000.0
            prompt_points, prompt_labels = empty_prompt()
            is_mask_from_points = False

        t_dec = time.time()
        dec = self._run_decoder(
            prompt_points,
            prompt_labels,
            fused_embed,
            enc["high_res_features0"],
            enc["high_res_features1"],
            output_device="session",
        )
        dec_ms = (time.time() - t_dec) * 1000.0

        t_mem = time.time()
        mem = self._run_memory_encoder(
            dec["pred_mask_high_res"],
            enc["current_vision_feat"],
            dec["object_score_logits"],
            is_mask_from_points=is_mask_from_points,
            output_device="session",
        )
        mem_ms = (time.time() - t_mem) * 1000.0

        # Only keep the high-res mask as a transient output for visualization/benchmarking.
        # Future frames only reuse memory features, memory positions, and object pointers.
        pred_mask_high_res = _to_numpy(dec["pred_mask_high_res"])
        stored_object_score_logits = None
        eff_iou_score = None
        if self.use_memory_selection:
            stored_object_score_logits = _to_numpy(dec["object_score_logits"])
            eff_iou_score = self._compute_eff_iou_score(dec, stored_object_score_logits)

        state = TrackerFrameState(
            maskmem_features=_to_numpy(mem["maskmem_features"]),
            maskmem_pos_enc=_to_numpy(mem["maskmem_pos_enc"]),
            obj_ptr=_to_numpy(dec["obj_ptr"]),
            object_score_logits=stored_object_score_logits,
            eff_iou_score=eff_iou_score,
        )
        if is_conditioning_frame:
            self.non_cond_states.pop(frame_idx, None)
            self.cond_states[frame_idx] = state
        else:
            self.cond_states.pop(frame_idx, None)
            self.non_cond_states[frame_idx] = state
            self._trim_non_cond_history(frame_idx)

        return FrameResult(
            frame_idx=frame_idx,
            info=info,
            state=state,
            mask_uint8=mask_to_uint8(pred_mask_high_res, info),
            timings=FrameTimings(
                prep_ms=prep_ms,
                enc_ms=enc_ms,
                attn_ms=attn_ms,
                dec_ms=dec_ms,
                mem_ms=mem_ms,
                total_ms=(time.time() - t_total) * 1000.0,
            ),
        )

    def _run_decoder(
        self,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        image_embed: np.ndarray | ort.OrtValue,
        high_res_0: np.ndarray | ort.OrtValue,
        high_res_1: np.ndarray | ort.OrtValue,
        *,
        output_device: str,
    ) -> dict[str, np.ndarray | ort.OrtValue]:
        if point_coords is None or point_labels is None:
            point_coords, point_labels = empty_prompt()

        feed = {
            "point_coords": _as_f32c(point_coords),
            "point_labels": _as_i32c(point_labels),
            "image_embed": image_embed,
            "high_res_feats_0": high_res_0,
            "high_res_feats_1": high_res_1,
        }
        return self._dec_runner.run(feed, output_device=output_device)

    def _run_memory_attention(
        self,
        current_vision_feat: np.ndarray | ort.OrtValue,
        current_vision_pos_embed: np.ndarray | ort.OrtValue,
        memory_obj_ptrs: np.ndarray | ort.OrtValue,
        memory_obj_tpos: np.ndarray | ort.OrtValue,
        memory_mask_feats: np.ndarray | ort.OrtValue,
        memory_mask_pos: np.ndarray | ort.OrtValue,
        memory_mask_tpos_idx: np.ndarray | ort.OrtValue,
        *,
        output_device: str,
    ) -> dict[str, np.ndarray | ort.OrtValue]:
        feed = {
            "current_vision_feat": current_vision_feat,
            "current_vision_pos_embed": current_vision_pos_embed,
            "memory_obj_ptrs": memory_obj_ptrs,
            "memory_obj_tpos": memory_obj_tpos,
            "memory_mask_feats": memory_mask_feats,
            "memory_mask_pos": memory_mask_pos,
            "memory_mask_tpos_idx": (
                memory_mask_tpos_idx
                if isinstance(memory_mask_tpos_idx, ort.OrtValue)
                else np.ascontiguousarray(np.asarray(memory_mask_tpos_idx, dtype=np.int64))
            ),
        }
        return self._mat_runner.run(feed, output_device=output_device)

    def _run_memory_encoder(
        self,
        pred_mask_high_res: np.ndarray | ort.OrtValue,
        current_vision_feat: np.ndarray | ort.OrtValue,
        object_score_logits: np.ndarray | ort.OrtValue,
        *,
        is_mask_from_points: bool,
        output_device: str,
    ) -> dict[str, np.ndarray | ort.OrtValue]:
        feed = {
            "pred_mask_high_res": pred_mask_high_res,
            "current_vision_feat": current_vision_feat,
            "object_score_logits": object_score_logits,
            "is_mask_from_points": np.ascontiguousarray(
                np.array([1.0 if is_mask_from_points else 0.0], dtype=np.float32)
            ),
        }
        return self._men_runner.run(feed, output_device=output_device)

    def _build_memory_input_buffers(self) -> dict[str, _InputBuffer]:
        if self.static_num_mem_frames is None and self.static_num_obj_ptrs is None:
            return {}

        buffers: dict[str, _InputBuffer] = {}
        if self.static_num_mem_frames is not None:
            buffers["memory_mask_feats"] = self._make_input_buffer(
                (self.static_num_mem_frames, 64, 72, 72),
                np.float32,
            )
            buffers["memory_mask_pos"] = self._make_input_buffer(
                (self.static_num_mem_frames, 64, 72, 72),
                np.float32,
            )
            buffers["memory_mask_tpos_idx"] = self._make_input_buffer(
                (self.static_num_mem_frames,),
                np.int64,
            )
        if self.static_num_obj_ptrs is not None:
            buffers["memory_obj_ptrs"] = self._make_input_buffer(
                (self.static_num_obj_ptrs, 256),
                np.float32,
            )
            buffers["memory_obj_tpos"] = self._make_input_buffer(
                (self.static_num_obj_ptrs,),
                np.float32,
            )
        return buffers

    def _resolve_warmup_enabled(self) -> bool:
        mode = os.getenv("SAM3_ORT_WARMUP", "auto").strip().lower()
        if mode in ("", "auto"):
            return self.device_type != "cpu"
        if mode in ("1", "true", "yes", "on"):
            return True
        if mode in ("0", "false", "no", "off"):
            return False
        raise SystemExit("SAM3_ORT_WARMUP must be auto, on, or off.")

    def _warmup_runtime(self) -> None:
        print("[INFO] Warming up ONNX tracker kernels...")
        try:
            self.reset()

            dummy_frame = np.zeros((1008, 1008, 3), dtype=np.uint8)
            prepared = self.prepare_frame(dummy_frame)
            warmup_points, warmup_labels = prepare_prompt_points(
                [(prepared.info.target_size // 2, prepared.info.target_size // 2)],
                [1],
                prepared.info,
            )

            # Warm the interactive preview path as well so the first click is responsive.
            self.preview_prompt_mask(prepared, warmup_points, warmup_labels)

            enc = prepared.encoder_outputs
            dec_cond = self._run_decoder(
                warmup_points,
                warmup_labels,
                enc["image_embeddings"],
                enc["high_res_features0"],
                enc["high_res_features1"],
                output_device="session",
            )
            mem_cond = self._run_memory_encoder(
                dec_cond["pred_mask_high_res"],
                enc["current_vision_feat"],
                dec_cond["object_score_logits"],
                is_mask_from_points=True,
                output_device="session",
            )
            self.cond_states[0] = TrackerFrameState(
                maskmem_features=_to_numpy(mem_cond["maskmem_features"]),
                maskmem_pos_enc=_to_numpy(mem_cond["maskmem_pos_enc"]),
                obj_ptr=_to_numpy(dec_cond["obj_ptr"]),
            )

            mem_inputs = self._select_memory_inputs(1)
            fused_embed = self._run_memory_attention(
                enc["current_vision_feat"],
                enc["current_vision_pos_embed"],
                mem_inputs["memory_obj_ptrs"],
                mem_inputs["memory_obj_tpos"],
                mem_inputs["memory_mask_feats"],
                mem_inputs["memory_mask_pos"],
                mem_inputs["memory_mask_tpos_idx"],
                output_device="session",
            )["fused_feat"]
            dec_track = self._run_decoder(
                *empty_prompt(),
                fused_embed,
                enc["high_res_features0"],
                enc["high_res_features1"],
                output_device="session",
            )
            self._run_memory_encoder(
                dec_track["pred_mask_high_res"],
                enc["current_vision_feat"],
                dec_track["object_score_logits"],
                is_mask_from_points=False,
                output_device="session",
            )
        except Exception as exc:
            print(f"[WARN] ONNX tracker warmup skipped: {exc}")
        finally:
            self.reset()

    def _make_input_buffer(self, shape: tuple[int, ...], dtype) -> _InputBuffer:
        array = np.zeros(shape, dtype=dtype)
        ortvalue = None
        if self.device_type != "cpu" and self.uses_iobinding:
            ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, self.device_type, 0)
            ortvalue.update_inplace(array)
        return _InputBuffer(array=array, ortvalue=ortvalue)

    def _frame_filter(self, frame_idx: int, r: int) -> list[int]:
        if frame_idx == 0:
            return []

        max_num = max(1, self.max_obj_ptrs)
        valid_indices: list[int] = []
        for prev_idx in range(frame_idx - 1, 0, -r):
            state = self.non_cond_states.get(prev_idx)
            if state is None or state.eff_iou_score is None:
                continue
            if state.eff_iou_score > self.mf_threshold:
                valid_indices.insert(0, prev_idx)
            if len(valid_indices) >= max_num - 1:
                break

        must_include = frame_idx - 1
        if must_include >= 0 and must_include not in valid_indices:
            valid_indices.append(must_include)
        return valid_indices

    def _max_non_cond_history(self) -> int:
        spatial_window = max(0, self.num_maskmem - 1) * max(1, self.memory_temporal_stride_for_eval)
        pointer_window = max(0, self.max_obj_ptrs - 1)
        if not self.use_memory_selection:
            return max(spatial_window, pointer_window)

        # Native trimming for memory-selection mode keeps a much longer tail for pointer scoring.
        return max(spatial_window, 20 * max(1, self.max_obj_ptrs))

    def _trim_non_cond_history(self, frame_idx: int) -> None:
        keep_window = self._max_non_cond_history()
        if keep_window <= 0:
            self.non_cond_states.clear()
            return

        min_frame_idx = frame_idx - keep_window
        stale_indices = [idx for idx in self.non_cond_states if idx < min_frame_idx]
        for idx in stale_indices:
            self.non_cond_states.pop(idx, None)

    def _compute_eff_iou_score(
        self,
        decoder_outputs: dict[str, np.ndarray | ort.OrtValue],
        object_score_logits: np.ndarray,
    ) -> float | None:
        logits = np.asarray(object_score_logits, dtype=np.float32)
        if logits.size == 0:
            return None

        object_score_norm = np.where(
            logits > 0.0,
            _sigmoid_np(logits) * 2.0 - 1.0,
            0.0,
        ).astype(np.float32, copy=False)

        raw_iou_scores = decoder_outputs.get("iou_scores")
        if raw_iou_scores is None:
            # Fallback for tracker bundles that do not expose IoU scores.
            return float(object_score_norm.mean())

        iou_scores = np.asarray(_to_numpy(raw_iou_scores), dtype=np.float32)
        if iou_scores.size == 0:
            return float(object_score_norm.mean())

        best_iou = np.max(iou_scores, axis=-1, keepdims=True).astype(np.float32, copy=False)
        return float(np.mean(object_score_norm * best_iou))

    def _pack_memory_inputs(
        self,
        name: str,
        value: np.ndarray,
        target: int | None,
    ) -> np.ndarray | ort.OrtValue:
        if target is None:
            return np.ascontiguousarray(value)

        buffer = self._memory_input_buffers[name]
        buffer.array.fill(0)
        count = min(int(value.shape[0]), int(target))
        if count > 0:
            buffer.array[:count] = value[:count]
        return buffer.sync()

    def _pack_memory_rows(
        self,
        name: str,
        rows: list[np.ndarray],
        target: int | None,
        *,
        dtype,
        empty_shape: tuple[int, ...],
    ) -> np.ndarray | ort.OrtValue:
        if target is None:
            if not rows:
                return np.zeros(empty_shape, dtype=dtype)
            return np.ascontiguousarray(np.stack(rows, axis=0).astype(dtype, copy=False))

        buffer = self._memory_input_buffers[name]
        buffer.array.fill(0)
        count = min(len(rows), int(target))
        if count > 0:
            for idx in range(count):
                buffer.array[idx] = rows[idx]
        return buffer.sync()

    @staticmethod
    def _memory_row(value: np.ndarray | ort.OrtValue, *, dtype) -> np.ndarray:
        row = np.asarray(_to_numpy(value))
        if row.ndim > 0 and row.shape[0] == 1:
            row = row[0]
        return np.ascontiguousarray(row.astype(dtype, copy=False))

    def _limit_spatial_memory_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        target = self.static_num_mem_frames
        if target is None or len(entries) <= target:
            return entries

        cond_entries = [entry for entry in entries if bool(entry["is_cond"])]
        non_cond_entries = [entry for entry in entries if not bool(entry["is_cond"])]

        selected_keys: set[tuple[bool, int]] = set()

        def entry_key(entry: dict[str, Any]) -> tuple[bool, int]:
            return bool(entry["is_cond"]), int(entry["frame_idx"])

        def add_entry(entry: dict[str, Any] | None) -> None:
            if entry is None or len(selected_keys) >= int(target):
                return
            selected_keys.add(entry_key(entry))

        latest_cond = (
            max(cond_entries, key=lambda entry: int(entry["frame_idx"])) if cond_entries else None
        )
        latest_non_cond = (
            max(non_cond_entries, key=lambda entry: int(entry["frame_idx"]))
            if non_cond_entries
            else None
        )
        first_cond = (
            min(cond_entries, key=lambda entry: int(entry["frame_idx"])) if cond_entries else None
        )

        # Preserve the most recent conditioning slice and the freshest dense memory first.
        add_entry(latest_cond)
        add_entry(latest_non_cond)
        if self.keep_first_cond_frame:
            add_entry(first_cond)

        for entry in sorted(cond_entries, key=lambda item: int(item["frame_idx"]), reverse=True):
            add_entry(entry)
        for entry in sorted(non_cond_entries, key=lambda item: int(item["frame_idx"]), reverse=True):
            add_entry(entry)

        limited_entries = [entry for entry in entries if entry_key(entry) in selected_keys]
        if limited_entries:
            return limited_entries[: int(target)]
        return entries[: int(target)]

    def _select_memory_inputs(self, frame_idx: int) -> dict[str, np.ndarray | ort.OrtValue]:
        if frame_idx <= 0:
            raise RuntimeError("Memory inputs are only defined for non-conditioning frames.")

        cond_outputs, unselected_cond_outputs = _select_closest_cond_frames(
            frame_idx,
            self.cond_states,
            self.max_cond_frames_in_attn,
            keep_first_cond_frame=self.keep_first_cond_frame,
        )

        spatial_entries: list[dict[str, Any]] = [
            {
                "frame_idx": int(cond_idx),
                "state": state,
                "tpos_idx": self.num_maskmem - 1,
                "is_cond": True,
            }
            for cond_idx, state in cond_outputs.items()
        ]
        r = max(1, int(self.memory_temporal_stride_for_eval))
        valid_indices = self._frame_filter(frame_idx, r) if self.use_memory_selection else []
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if self.use_memory_selection:
                if t_rel > len(valid_indices):
                    continue
                prev_frame_idx = valid_indices[-t_rel]
            else:
                if t_rel == 1:
                    prev_frame_idx = frame_idx - 1
                else:
                    prev_frame_idx = ((frame_idx - 2) // r) * r
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
            if prev_frame_idx < 0:
                continue
            state = self.non_cond_states.get(prev_frame_idx)
            if state is None:
                state = unselected_cond_outputs.get(prev_frame_idx)
            if state is None:
                continue
            spatial_entries.append(
                {
                    "frame_idx": int(prev_frame_idx),
                    "state": state,
                    "tpos_idx": self.num_maskmem - t_pos - 1,
                    "is_cond": False,
                }
            )

        if not spatial_entries:
            raise RuntimeError("No spatial memory was available for memory attention.")

        spatial_entries = self._limit_spatial_memory_entries(spatial_entries)

        memory_mask_feats = self._pack_memory_rows(
            "memory_mask_feats",
            [
                self._memory_row(entry["state"].maskmem_features, dtype=np.float32)
                for entry in spatial_entries
            ],
            self.static_num_mem_frames,
            dtype=np.float32,
            empty_shape=(0, 64, 72, 72),
        )
        memory_mask_pos = self._pack_memory_rows(
            "memory_mask_pos",
            [
                self._memory_row(entry["state"].maskmem_pos_enc, dtype=np.float32)
                for entry in spatial_entries
            ],
            self.static_num_mem_frames,
            dtype=np.float32,
            empty_shape=(0, 64, 72, 72),
        )
        memory_mask_tpos_idx = np.asarray(
            [int(entry["tpos_idx"]) for entry in spatial_entries],
            dtype=np.int64,
        )

        pointer_items: list[tuple[TrackerFrameState, float]] = [
            (state, float(frame_idx - cond_idx))
            for cond_idx, state in cond_outputs.items()
            if cond_idx <= frame_idx
        ]
        if self.use_memory_selection:
            pointer_indices = list(reversed(valid_indices))
        else:
            pointer_indices = list(range(frame_idx - 1, max(-1, frame_idx - self.max_obj_ptrs), -1))
        for t_diff, prev_frame_idx in enumerate(pointer_indices, start=1):
            if prev_frame_idx < 0:
                break
            state = self.non_cond_states.get(prev_frame_idx)
            if state is None:
                state = unselected_cond_outputs.get(prev_frame_idx)
            if state is None:
                continue
            pointer_items.append((state, float(t_diff)))

        if pointer_items:
            memory_obj_ptrs = self._pack_memory_rows(
                "memory_obj_ptrs",
                [self._memory_row(item[0].obj_ptr, dtype=np.float32) for item in pointer_items],
                self.static_num_obj_ptrs,
                dtype=np.float32,
                empty_shape=(0, 256),
            )
            memory_obj_tpos = np.asarray([item[1] for item in pointer_items], dtype=np.float32)
        else:
            memory_obj_ptrs = self._pack_memory_rows(
                "memory_obj_ptrs",
                [],
                self.static_num_obj_ptrs,
                dtype=np.float32,
                empty_shape=(0, 256),
            )
            memory_obj_tpos = np.zeros((0,), np.float32)

        return {
            "memory_obj_ptrs": memory_obj_ptrs,
            "memory_obj_tpos": self._pack_memory_inputs(
                "memory_obj_tpos",
                memory_obj_tpos,
                self.static_num_obj_ptrs,
            ),
            "memory_mask_feats": memory_mask_feats,
            "memory_mask_pos": memory_mask_pos,
            "memory_mask_tpos_idx": self._pack_memory_inputs(
                "memory_mask_tpos_idx",
                memory_mask_tpos_idx,
                self.static_num_mem_frames,
            ),
        }

    @staticmethod
    def _profiled_name(base_name: str, graph_profile: str) -> str:
        if not graph_profile or graph_profile == "default":
            return base_name
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        return f"{stem}_{graph_profile}{suffix}"

    @classmethod
    def _resolve_graph_profile(cls, requested_max_mem_frames: int) -> str:
        return (
            MULTI_GRAPH_PROFILE
            if int(requested_max_mem_frames) > DEFAULT_MAX_MEM_FRAMES
            else SINGLE_GRAPH_PROFILE
        )

    @staticmethod
    def _preferred_tracker_precisions() -> tuple[str, ...]:
        requested_precision = os.getenv("SAM3_ORT_TRACKER_PRECISION", "auto").strip().lower()
        if requested_precision in ("", "auto"):
            accel = os.getenv("SAM3_ORT_ACCEL", "auto").strip().lower()
            prefers_fp16 = accel != "cpu" and any(
                provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
                for provider in ort.get_available_providers()
            )
            return ("fp16", "fp32") if prefers_fp16 else ("fp32", "fp16")
        if requested_precision == "fp16":
            return ("fp16", "fp32")
        if requested_precision == "fp32":
            return ("fp32", "fp16")
        raise SystemExit(
            "SAM3_ORT_TRACKER_PRECISION must be auto, fp16, or fp32."
        )

    @classmethod
    def _profiled_precision_name(
        cls,
        base_name: str,
        graph_profile: str,
        precision: str,
    ) -> str:
        name = cls._profiled_name(base_name, graph_profile)
        if precision != "fp16":
            return name
        stem = Path(name).stem
        suffix = Path(name).suffix
        return f"{stem}_fp16{suffix}"

    @classmethod
    def _resolve_tracker_bundle(
        cls,
        onnx_dir: Path,
        requested_graph_profile: str,
    ) -> _ResolvedTrackerBundle:
        base_names = {
            "constants_path": "video_constants.npz",
            "decoder_path": "image_decoder.onnx",
            "memory_attention_path": "memory_attention.onnx",
            "memory_encoder_path": "memory_encoder.onnx",
        }
        last_candidates: dict[str, Path] = {
            key: onnx_dir / value for key, value in base_names.items()
        }

        for precision in cls._preferred_tracker_precisions():
            resolved = {
                key: onnx_dir
                / cls._profiled_precision_name(value, requested_graph_profile, precision)
                for key, value in base_names.items()
            }
            last_candidates = resolved
            if all(path.exists() for path in resolved.values()):
                return _ResolvedTrackerBundle(
                    graph_profile=requested_graph_profile,
                    precision=precision,
                    constants_path=resolved["constants_path"],
                    decoder_path=resolved["decoder_path"],
                    memory_attention_path=resolved["memory_attention_path"],
                    memory_encoder_path=resolved["memory_encoder_path"],
                )

        raise SystemExit(
            "Missing a complete tracker ONNX bundle for the requested runtime mode. Expected matching "
            "decoder, memory attention, memory encoder, and video constants files like "
            f"{last_candidates['decoder_path'].name} in the video export directory. "
            "Run export\\onnx_export.py first to regenerate the tracker graphs."
        )

    @classmethod
    def _resolve_encoder_path(cls, onnx_dir: Path) -> Path:
        custom_candidates = [
            onnx_dir / "image_encoder.onnx",
            onnx_dir / "image_encoder_fp16.onnx",
        ]
        for custom_encoder in custom_candidates:
            if custom_encoder.exists():
                return custom_encoder

        repo_root = Path(__file__).resolve().parent.parent
        shared_dir = repo_root / "checkpoints" / "sam3" / "onnx"

        accel = os.getenv("SAM3_ORT_ACCEL", "auto").strip().lower()
        prefers_fp16 = accel != "cpu" and any(
            provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
            for provider in ort.get_available_providers()
        )
        encoder_candidates = (
            [shared_dir / "vision_encoder_fp16.onnx", shared_dir / "vision_encoder.onnx"]
            if prefers_fp16
            else [shared_dir / "vision_encoder.onnx", shared_dir / "vision_encoder_fp16.onnx"]
        )

        seen = set()
        for candidate in encoder_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            sidecar = candidate.with_name(candidate.name + "_data")
            if candidate.exists() and sidecar.exists():
                return candidate

        raise SystemExit(
            "Could not find a complete encoder ONNX pair. Expected image_encoder.onnx in the "
            "video export dir or vision_encoder*.onnx plus .onnx_data under checkpoints/sam3/onnx."
        )

    @classmethod
    def _load_video_constants(cls, constants_path: Path) -> dict[str, np.ndarray]:
        if not constants_path.exists():
            raise SystemExit(
                f"Missing {constants_path}. Run export\\onnx_export.py first to generate the video constants bundle."
            )

        with np.load(constants_path) as data:
            return {key: np.ascontiguousarray(data[key]) for key in data.files}

    @staticmethod
    def _fixed_input_dim(session, input_name: str, axis: int) -> int | None:
        for model_input in session.get_inputs():
            if model_input.name != input_name:
                continue
            shape = model_input.shape
            if axis >= len(shape):
                return None
            dim = shape[axis]
            if isinstance(dim, int) and dim > 0:
                return int(dim)
            return None
        return None

    @staticmethod
    def _fit_first_dim(arr: np.ndarray, target: int) -> np.ndarray:
        current = int(arr.shape[0])
        if current == target:
            return np.ascontiguousarray(arr)
        if current > target:
            return np.ascontiguousarray(arr[:target])

        pad_shape = (target - current, *arr.shape[1:])
        pad = np.zeros(pad_shape, dtype=arr.dtype)
        return np.ascontiguousarray(np.concatenate((arr, pad), axis=0))
