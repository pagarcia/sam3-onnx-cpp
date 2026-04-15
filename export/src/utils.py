# sam3-onnx-cpp/export/src/utils.py
import os

import torch

OPSET = 18
OPTIMIZE = False
RUN_ONNX_CHECKER = False


def _maybe_check(path: str, label: str) -> None:
    if RUN_ONNX_CHECKER:
        import onnx

        model = onnx.load(path)
        onnx.checker.check_model(model)
    print(f"[INFO] Exported {label}: {path}")


def export_image_encoder(model, outdir: str) -> None:
    raise RuntimeError(
        "SAM3 image encoder export is intentionally disabled here. "
        "The SAM3 backbone still hits unsupported complex rotary ops during ONNX export, "
        "so the video path reuses the shipped vision_encoder*.onnx and only exports the "
        "tracker-specific decoder/memory modules plus video_constants.npz."
    )


def export_image_decoder(model, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "image_decoder.onnx")

    point_coords = torch.randn(1, 2, 2, dtype=torch.float32)
    point_labels = torch.tensor([[1, 0]], dtype=torch.int32)
    image_embed = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    high_res_0 = torch.randn(1, 32, 288, 288, dtype=torch.float32)
    high_res_1 = torch.randn(1, 64, 144, 144, dtype=torch.float32)

    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, high_res_0, high_res_1),
        path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
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
        ],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
    )
    _maybe_check(path, "image decoder")


def export_memory_attention(model, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "memory_attention.onnx")

    current_vision_feat = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    current_vision_pos = torch.randn(72 * 72, 1, 256, dtype=torch.float32)
    memory_obj_ptrs = torch.randn(1, 256, dtype=torch.float32)
    memory_obj_tpos = torch.tensor([1.0], dtype=torch.float32)
    memory_mask_feats = torch.randn(1, 64, 72, 72, dtype=torch.float32)
    memory_mask_pos = torch.randn(1, 64, 72, 72, dtype=torch.float32)
    memory_mask_tpos_idx = torch.tensor([6], dtype=torch.int64)

    torch.onnx.export(
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
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
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
        dynamic_axes={
            "memory_obj_ptrs": {0: "num_obj_ptrs"},
            "memory_obj_tpos": {0: "num_obj_ptrs"},
            "memory_mask_feats": {0: "num_mem_frames"},
            "memory_mask_pos": {0: "num_mem_frames"},
            "memory_mask_tpos_idx": {0: "num_mem_frames"},
        },
    )
    _maybe_check(path, "memory attention")


def export_memory_encoder(model, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "memory_encoder.onnx")

    pred_mask_high_res = torch.randn(1, 1, 1008, 1008, dtype=torch.float32)
    current_vision_feat = torch.randn(1, 256, 72, 72, dtype=torch.float32)
    object_score_logits = torch.randn(1, 1, dtype=torch.float32)
    is_mask_from_points = torch.tensor([1.0], dtype=torch.float32)

    torch.onnx.export(
        model,
        (
            pred_mask_high_res,
            current_vision_feat,
            object_score_logits,
            is_mask_from_points,
        ),
        path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "pred_mask_high_res",
            "current_vision_feat",
            "object_score_logits",
            "is_mask_from_points",
        ],
        output_names=["maskmem_features", "maskmem_pos_enc"],
    )
    _maybe_check(path, "memory encoder")
