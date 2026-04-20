#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from onnx_test_utils import PrepInfo, green_overlay, make_session, preprocess_image_bgr


DEFAULT_NUM_MASKMEM = 7
DEFAULT_MAX_OBJ_PTRS = 16
FAST_DEFAULT_MAX_MEM_FRAMES = 2
FAST_DEFAULT_MAX_OBJ_PTRS = 16


def _as_f32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _as_i32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.int32, copy=False))


def _to_numpy(value: np.ndarray | ort.OrtValue) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.ascontiguousarray(value.numpy())


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


def mask_to_uint8(mask_logits_high_res: np.ndarray, info: PrepInfo) -> np.ndarray:
    mask_logits = mask_logits_high_res[0, 0]
    mask_resized = cv2.resize(
        mask_logits,
        (info.orig_hw[1], info.orig_hw[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return (mask_resized > 0.0).astype(np.uint8) * 255


def mask_to_overlay(frame_bgr: np.ndarray, mask_logits_high_res: np.ndarray, info: PrepInfo) -> np.ndarray:
    mask_uint8 = mask_to_uint8(mask_logits_high_res, info)
    return green_overlay(frame_bgr, mask_uint8, alpha=0.5)


def normalize_encoder_outputs(
    raw_outputs: dict[str, np.ndarray],
    constants: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if "image_embeddings" in raw_outputs:
        return raw_outputs

    current_vision_feat = raw_outputs["image_embeddings.2"]
    return {
        "image_embeddings": current_vision_feat + constants["no_mem_embed_bchw"],
        "high_res_features0": raw_outputs["image_embeddings.0"],
        "high_res_features1": raw_outputs["image_embeddings.1"],
        "current_vision_feat": current_vision_feat,
        "current_vision_pos_embed": constants["current_vision_pos_embed"],
    }


@dataclass(frozen=True)
class PreparedFrame:
    encoder_outputs: dict[str, np.ndarray]
    info: PrepInfo


@dataclass(frozen=True)
class TrackerFrameState:
    maskmem_features: np.ndarray
    maskmem_pos_enc: np.ndarray
    obj_ptr: np.ndarray
    pred_mask_high_res: np.ndarray
    object_score_logits: np.ndarray


@dataclass(frozen=True)
class FrameTimings:
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
        for output_name in output_names:
            binding.bind_output(output_name, bind_device, 0)

        self.session.run_with_iobinding(binding)
        if bind_device == "cpu":
            values = binding.copy_outputs_to_cpu()
            return dict(zip(output_names, values))

        binding.synchronize_outputs()
        values = list(binding.get_outputs())
        return dict(zip(output_names, values))


class Sam3OnnxTrackerSession:
    def __init__(
        self,
        onnx_dir: Path,
        *,
        safe: bool = False,
        max_mem_frames: int = FAST_DEFAULT_MAX_MEM_FRAMES,
        max_obj_ptrs: int = FAST_DEFAULT_MAX_OBJ_PTRS,
        variant: str = "",
    ) -> None:
        self.onnx_dir = Path(onnx_dir).resolve()
        self.variant = variant.strip()
        self.constants = self._load_video_constants(self.onnx_dir, self.variant)

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

        enc_path = self._resolve_encoder_path(self.onnx_dir, self.variant)
        dec_path = self._resolve_model_path(self.onnx_dir, "image_decoder.onnx", self.variant)
        mat_path = self._resolve_model_path(self.onnx_dir, "memory_attention.onnx", self.variant)
        men_path = self._resolve_model_path(self.onnx_dir, "memory_encoder.onnx", self.variant)

        for path in (enc_path, dec_path, mat_path, men_path):
            if not path.exists():
                raise SystemExit(f"Missing ONNX file: {path}")

        self.sess_enc = make_session(str(enc_path), tag="video_image_encoder", safe=safe)
        self.sess_dec = make_session(str(dec_path), tag="video_image_decoder", safe=safe)
        self.sess_mat = make_session(str(mat_path), tag="video_memory_attention", safe=safe)
        self.sess_men = make_session(str(men_path), tag="video_memory_encoder", safe=safe)

        self._dec_runner = _OrtSessionRunner(self.sess_dec, "video_image_decoder")
        self._mat_runner = _OrtSessionRunner(self.sess_mat, "video_memory_attention")
        self._men_runner = _OrtSessionRunner(self.sess_men, "video_memory_encoder")

        self.static_num_mem_frames = self._fixed_input_dim(self.sess_mat, "memory_mask_feats", 0)
        self.static_num_obj_ptrs = self._fixed_input_dim(self.sess_mat, "memory_obj_ptrs", 0)
        self.num_maskmem = (
            int(self.static_num_mem_frames)
            if self.static_num_mem_frames is not None
            else requested_num_maskmem
        )
        self.max_obj_ptrs = (
            int(self.static_num_obj_ptrs)
            if self.static_num_obj_ptrs is not None
            else requested_max_obj_ptrs
        )

        self.reset()

    @property
    def uses_iobinding(self) -> bool:
        return (
            self._dec_runner.enable_iobinding
            or self._mat_runner.enable_iobinding
            or self._men_runner.enable_iobinding
        )

    def reset(self) -> None:
        self.cond_states: dict[int, TrackerFrameState] = {}
        self.non_cond_states: dict[int, TrackerFrameState] = {}

    def prepare_frame(self, frame_bgr: np.ndarray) -> PreparedFrame:
        pixel_values, info = preprocess_image_bgr(frame_bgr, target_size=1008)
        input_name = self.sess_enc.get_inputs()[0].name
        output_names = [output.name for output in self.sess_enc.get_outputs()]
        values = self.sess_enc.run(None, {input_name: _as_f32c(pixel_values)})
        raw_outputs = dict(zip(output_names, values))
        return PreparedFrame(
            encoder_outputs=normalize_encoder_outputs(raw_outputs, self.constants),
            info=info,
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
        dec = self._run_decoder(
            prompt_points,
            prompt_labels,
            prepared.encoder_outputs["image_embeddings"],
            prepared.encoder_outputs["high_res_features0"],
            prepared.encoder_outputs["high_res_features1"],
            output_device="cpu",
        )
        return dec["pred_mask_high_res"]

    def process_frame(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        *,
        prepared: PreparedFrame | None = None,
        prompt_points: np.ndarray | None = None,
        prompt_labels: np.ndarray | None = None,
    ) -> FrameResult:
        t_total = time.time()

        if frame_idx == 0:
            if prepared is None:
                prepared = self.prepare_frame(frame_bgr)
            enc = prepared.encoder_outputs
            info = prepared.info
            fused_embed = enc["image_embeddings"]
            prompt_points = prompt_points if prompt_points is not None else empty_prompt()[0]
            prompt_labels = prompt_labels if prompt_labels is not None else empty_prompt()[1]
            is_mask_from_points = True
            attn_ms = 0.0
            enc_ms = 0.0
        else:
            t_enc = time.time()
            prepared = self.prepare_frame(frame_bgr)
            enc = prepared.encoder_outputs
            info = prepared.info
            enc_ms = (time.time() - t_enc) * 1000.0

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

        state = TrackerFrameState(
            maskmem_features=_to_numpy(mem["maskmem_features"]),
            maskmem_pos_enc=_to_numpy(mem["maskmem_pos_enc"]),
            obj_ptr=_to_numpy(dec["obj_ptr"]),
            pred_mask_high_res=_to_numpy(dec["pred_mask_high_res"]),
            object_score_logits=_to_numpy(dec["object_score_logits"]),
        )
        if frame_idx == 0:
            self.cond_states[frame_idx] = state
        else:
            self.non_cond_states[frame_idx] = state

        return FrameResult(
            frame_idx=frame_idx,
            info=info,
            state=state,
            mask_uint8=mask_to_uint8(state.pred_mask_high_res, info),
            timings=FrameTimings(
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
            "memory_mask_tpos_idx": np.ascontiguousarray(
                _to_numpy(memory_mask_tpos_idx).astype(np.int64, copy=False)
                if isinstance(memory_mask_tpos_idx, ort.OrtValue)
                else np.asarray(memory_mask_tpos_idx, dtype=np.int64)
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

    def _select_memory_inputs(self, frame_idx: int) -> dict[str, np.ndarray]:
        spatial_items = []
        if 0 in self.cond_states and frame_idx > 0:
            spatial_items.append((self.cond_states[0], self.num_maskmem - 1))

        for t_pos in range(1, self.num_maskmem):
            prev_idx = frame_idx - (self.num_maskmem - t_pos)
            if prev_idx < 0:
                continue
            state = self.non_cond_states.get(prev_idx)
            if state is None:
                continue
            spatial_items.append((state, self.num_maskmem - t_pos - 1))

        if not spatial_items:
            raise RuntimeError("No spatial memory was available for memory attention.")

        memory_mask_feats = np.concatenate(
            [item[0].maskmem_features for item in spatial_items], axis=0
        )
        memory_mask_pos = np.concatenate(
            [item[0].maskmem_pos_enc for item in spatial_items], axis=0
        )
        memory_mask_tpos_idx = np.array([item[1] for item in spatial_items], dtype=np.int64)

        pointer_items = []
        if 0 in self.cond_states and frame_idx > 0:
            pointer_items.append((self.cond_states[0], float(frame_idx)))
        for t_diff in range(1, self.max_obj_ptrs):
            prev_idx = frame_idx - t_diff
            if prev_idx < 0:
                break
            state = self.non_cond_states.get(prev_idx)
            if state is not None:
                pointer_items.append((state, float(t_diff)))

        if pointer_items:
            memory_obj_ptrs = np.concatenate([item[0].obj_ptr for item in pointer_items], axis=0)
            memory_obj_tpos = np.array([item[1] for item in pointer_items], dtype=np.float32)
        else:
            memory_obj_ptrs = np.zeros((0, 256), np.float32)
            memory_obj_tpos = np.zeros((0,), np.float32)

        if self.static_num_mem_frames is not None:
            memory_mask_feats = self._fit_first_dim(memory_mask_feats, self.static_num_mem_frames)
            memory_mask_pos = self._fit_first_dim(memory_mask_pos, self.static_num_mem_frames)
            memory_mask_tpos_idx = self._fit_first_dim(memory_mask_tpos_idx, self.static_num_mem_frames)
        if self.static_num_obj_ptrs is not None:
            memory_obj_ptrs = self._fit_first_dim(memory_obj_ptrs, self.static_num_obj_ptrs)
            memory_obj_tpos = self._fit_first_dim(memory_obj_tpos, self.static_num_obj_ptrs)

        return {
            "memory_obj_ptrs": memory_obj_ptrs,
            "memory_obj_tpos": memory_obj_tpos,
            "memory_mask_feats": memory_mask_feats,
            "memory_mask_pos": memory_mask_pos,
            "memory_mask_tpos_idx": memory_mask_tpos_idx,
        }

    @staticmethod
    def _variant_name(base_name: str, variant: str) -> str:
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        return f"{stem}_{variant}{suffix}" if variant else base_name

    @classmethod
    def _resolve_model_path(cls, onnx_dir: Path, base_name: str, variant: str) -> Path:
        candidate = onnx_dir / cls._variant_name(base_name, variant)
        if candidate.exists():
            return candidate
        fallback = onnx_dir / base_name
        if fallback.exists():
            return fallback
        return candidate

    @classmethod
    def _resolve_encoder_path(cls, onnx_dir: Path, variant: str) -> Path:
        custom_encoder = cls._resolve_model_path(onnx_dir, "image_encoder.onnx", variant)
        if custom_encoder.exists():
            return custom_encoder

        repo_root = Path(__file__).resolve().parent.parent
        shared_dir = repo_root / "checkpoints" / "sam3" / "onnx"

        encoder_candidates = []
        wants_fp16 = "fp16" in variant.lower()
        wants_fp32 = "fp32" in variant.lower()
        if variant:
            encoder_candidates.extend(
                [
                    shared_dir / cls._variant_name("vision_encoder.onnx", variant),
                    shared_dir / cls._variant_name("vision_encoder_fp16.onnx", variant),
                ]
            )
            if wants_fp32:
                encoder_candidates.append(shared_dir / "vision_encoder.onnx")
            if wants_fp16:
                encoder_candidates.append(shared_dir / "vision_encoder_fp16.onnx")

        if wants_fp16:
            encoder_candidates.extend(
                [
                    shared_dir / "vision_encoder_fp16.onnx",
                    shared_dir / "vision_encoder.onnx",
                ]
            )
        else:
            encoder_candidates.extend(
                [
                    shared_dir / "vision_encoder.onnx",
                    shared_dir / "vision_encoder_fp16.onnx",
                ]
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
    def _load_video_constants(cls, onnx_dir: Path, variant: str) -> dict[str, np.ndarray]:
        constants_path = cls._resolve_model_path(onnx_dir, "video_constants.npz", variant)
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
