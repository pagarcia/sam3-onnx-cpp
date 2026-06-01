import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

OPSET = 18
OPTIMIZE = False
RUN_ONNX_CHECKER = False


@dataclass(frozen=True)
class ExportVariant:
    name: str
    precision: str = "fp32"
    max_mem_frames: int | None = None
    max_obj_ptrs: int | None = None
    static_memory_shapes: bool = False

    @property
    def token(self) -> str:
        parts = []
        if self.name:
            parts.append(self.name)
        if self.precision == "fp16":
            parts.append("fp16")
        return "_".join(parts)

    def filename(self, base_name: str) -> str:
        path = Path(base_name)
        if not self.token:
            return base_name
        return f"{path.stem}_{self.token}{path.suffix}"

    def label(self, base_label: str) -> str:
        if not self.token:
            return base_label
        return f"{base_label} [{self.token}]"


def _maybe_check(path: str, label: str) -> None:
    if RUN_ONNX_CHECKER:
        import onnx

        model = onnx.load(path)
        onnx.checker.check_model(model)
    print(f"[INFO] Exported {label}: {path}")


def _convert_to_fp16(src_path: str, dst_path: str, label: str) -> None:
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RuntimeError(
            "FP16 export requires both onnx and onnxruntime.transformers.float16."
        ) from exc

    model = onnx.load(src_path)
    model = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save_model(model, dst_path)
    _maybe_check(dst_path, label)


def _export_model(
    model,
    model_args: tuple,
    dst_path: str,
    *,
    label: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None,
    variant: ExportVariant,
) -> None:
    export_path = dst_path
    temp_path = None
    sidecar_path = f"{dst_path}.data"
    if os.path.exists(sidecar_path):
        os.remove(sidecar_path)
    if variant.precision == "fp16":
        with tempfile.NamedTemporaryFile(
            prefix="sam3_export_",
            suffix=".onnx",
            delete=False,
            dir=str(Path(dst_path).resolve().parent),
        ) as tmp:
            temp_path = tmp.name
            export_path = temp_path

    try:
        torch.onnx.export(
            model,
            model_args,
            export_path,
            export_params=True,
            opset_version=OPSET,
            optimize=OPTIMIZE,
            dynamo=False,
            external_data=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        if variant.precision == "fp16":
            _convert_to_fp16(export_path, dst_path, label)
        else:
            _maybe_check(dst_path, label)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if temp_path:
            temp_sidecar_path = f"{temp_path}.data"
            if os.path.exists(temp_sidecar_path):
                os.remove(temp_sidecar_path)


def export_image_encoder(model, outdir: str) -> None:
    raise RuntimeError(
        "SAM3 image encoder export is intentionally disabled here. "
        "The SAM3 backbone still hits unsupported complex rotary ops during ONNX export, "
        "so the video path reuses the shipped vision_encoder*.onnx and only exports the "
        "tracker-specific decoder/memory modules plus video_constants.npz."
    )


def export_image_decoder(model, outdir: str, variant: ExportVariant) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, variant.filename("image_decoder.onnx"))

    point_coords = torch.randn(1, 2, 2, dtype=torch.float32)
    point_labels = torch.tensor([[1, 0]], dtype=torch.int32)
    image_embed = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    high_res_0 = torch.randn(1, 32, 288, 288, dtype=torch.float32)
    high_res_1 = torch.randn(1, 64, 144, 144, dtype=torch.float32)

    _export_model(
        model,
        (point_coords, point_labels, image_embed, high_res_0, high_res_1),
        path,
        label=variant.label("image decoder"),
        input_names=[
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
        ],
        output_names=[
            "obj_ptr",
            "pred_mask",
            "pred_mask_high_res",
            "object_score_logits",
            "iou_scores",
            "pred_multimasks",
            "pred_multimasks_high_res",
        ],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        variant=variant,
    )


def export_image_decoder_mask(model, outdir: str, variant: ExportVariant) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, variant.filename("image_decoder_mask.onnx"))

    point_coords = torch.randn(1, 2, 2, dtype=torch.float32)
    point_labels = torch.tensor([[1, 0]], dtype=torch.int32)
    image_embed = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    high_res_0 = torch.randn(1, 32, 288, 288, dtype=torch.float32)
    high_res_1 = torch.randn(1, 64, 144, 144, dtype=torch.float32)
    mask_inputs = torch.randn(1, 1, 288, 288, dtype=torch.float32)

    _export_model(
        model,
        (point_coords, point_labels, image_embed, high_res_0, high_res_1, mask_inputs),
        path,
        label=variant.label("image decoder mask"),
        input_names=[
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
            "mask_inputs",
        ],
        output_names=[
            "obj_ptr",
            "pred_mask",
            "pred_mask_high_res",
            "object_score_logits",
            "iou_scores",
            "pred_multimasks",
            "pred_multimasks_high_res",
        ],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        variant=variant,
    )


def export_memory_attention(model, outdir: str, variant: ExportVariant) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, variant.filename("memory_attention.onnx"))

    num_mem_frames = int(variant.max_mem_frames or 1)
    num_obj_ptrs = int(variant.max_obj_ptrs or 1)

    current_vision_feat = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    current_vision_pos = torch.randn(72 * 72, 1, 256, dtype=torch.float32)
    memory_obj_ptrs = torch.randn(num_obj_ptrs, 256, dtype=torch.float32)
    memory_obj_tpos = torch.tensor([float(idx + 1) for idx in range(num_obj_ptrs)], dtype=torch.float32)
    memory_mask_feats = torch.randn(num_mem_frames, 64, 72, 72, dtype=torch.float32)
    memory_mask_pos = torch.randn(num_mem_frames, 64, 72, 72, dtype=torch.float32)
    memory_mask_tpos_idx = torch.tensor(
        [max(num_mem_frames - idx - 1, 0) for idx in range(num_mem_frames)],
        dtype=torch.int64,
    )

    dynamic_axes = None
    if not variant.static_memory_shapes:
        dynamic_axes = {
            "memory_obj_ptrs": {0: "num_obj_ptrs"},
            "memory_obj_tpos": {0: "num_obj_ptrs"},
            "memory_mask_feats": {0: "num_mem_frames"},
            "memory_mask_pos": {0: "num_mem_frames"},
            "memory_mask_tpos_idx": {0: "num_mem_frames"},
        }

    _export_model(
        model,
        (
            current_vision_feat,
            current_vision_pos,
            memory_obj_ptrs,
            memory_obj_tpos,
            memory_mask_feats,
            memory_mask_pos,
            memory_mask_tpos_idx,
        ),
        path,
        label=variant.label("memory attention"),
        input_names=[
            "current_vision_feat",
            "current_vision_pos_embed",
            "memory_obj_ptrs",
            "memory_obj_tpos",
            "memory_mask_feats",
            "memory_mask_pos",
            "memory_mask_tpos_idx",
        ],
        output_names=["fused_feat"],
        dynamic_axes=dynamic_axes,
        variant=variant,
    )


def export_memory_encoder(model, outdir: str, variant: ExportVariant) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, variant.filename("memory_encoder.onnx"))

    pred_mask_high_res = torch.randn(1, 1, 1008, 1008, dtype=torch.float32)
    current_vision_feat = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    object_score_logits = torch.randn(1, 1, dtype=torch.float32)
    is_mask_from_points = torch.tensor([1.0], dtype=torch.float32)

    _export_model(
        model,
        (
            pred_mask_high_res,
            current_vision_feat,
            object_score_logits,
            is_mask_from_points,
        ),
        path,
        label=variant.label("memory encoder"),
        input_names=[
            "pred_mask_high_res",
            "current_vision_feat",
            "object_score_logits",
            "is_mask_from_points",
        ],
        output_names=["maskmem_features", "maskmem_pos_enc"],
        dynamic_axes=None,
        variant=variant,
    )
