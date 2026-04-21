# sam3-onnx-cpp/export/onnx_export.py
import argparse
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _export_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_sam3_repo() -> Path:
    return _repo_root().parent / "sam3"


def _add_import_paths(sam3_repo: Path) -> None:
    sys.path.insert(0, str(sam3_repo.resolve()))
    sys.path.insert(0, str(_export_dir().resolve()))
    sys.path.insert(0, str(_repo_root().resolve()))


def _install_optional_sam3_stubs() -> None:
    if "sam3.model.edt" in sys.modules:
        edt_installed = True
    else:
        edt_installed = False

    if not edt_installed:
        edt_module = types.ModuleType("sam3.model.edt")

        def edt_triton(data: torch.Tensor) -> torch.Tensor:
            import cv2

            if data.dim() != 3:
                raise ValueError(f"Expected [B,H,W] tensor, got shape {tuple(data.shape)}")

            device = data.device
            data_np = data.detach().to("cpu").numpy()
            out = np.zeros(data_np.shape, dtype=np.float32)
            for idx, mask in enumerate(data_np):
                dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
                out[idx] = dist.astype(np.float32, copy=False)
            return torch.from_numpy(out).to(device=device)

        edt_module.edt_triton = edt_triton
        sys.modules["sam3.model.edt"] = edt_module

    if "sam3.train.data.collator" not in sys.modules:
        collator_module = types.ModuleType("sam3.train.data.collator")

        @dataclass
        class BatchedDatapoint:
            img_batch: torch.Tensor | None = None
            find_text_batch: list[str] | None = None
            find_inputs: list | None = None
            find_targets: list | None = None
            find_metadatas: list | None = None
            raw_images: list | None = None

        collator_module.BatchedDatapoint = BatchedDatapoint
        sys.modules["sam3.train.data.collator"] = collator_module

    if "sam3.model.sam3_video_inference" not in sys.modules:
        video_inference_module = types.ModuleType("sam3.model.sam3_video_inference")

        class Sam3VideoInferenceWithInstanceInteractivity:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Video inference stubs are loaded for ONNX export only and should not be instantiated."
                )

        video_inference_module.Sam3VideoInferenceWithInstanceInteractivity = (
            Sam3VideoInferenceWithInstanceInteractivity
        )
        sys.modules["sam3.model.sam3_video_inference"] = video_inference_module

    if "sam3.model.sam3_video_predictor" not in sys.modules:
        video_predictor_module = types.ModuleType("sam3.model.sam3_video_predictor")

        class Sam3VideoPredictorMultiGPU:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Video predictor stubs are loaded for ONNX export only and should not be instantiated."
                )

        video_predictor_module.Sam3VideoPredictorMultiGPU = Sam3VideoPredictorMultiGPU
        sys.modules["sam3.model.sam3_video_predictor"] = video_predictor_module


def _build_model(args):
    from sam3.model_builder import build_sam3_image_model

    if not args.checkpoint and not args.load_from_hf:
        raise SystemExit(
            "Provide --checkpoint or pass --load-from-hf so the exporter does not run on random weights."
        )

    build_kwargs = {
        "device": "cpu",
        "eval_mode": True,
        "enable_inst_interactivity": True,
        "load_from_HF": bool(args.load_from_hf),
    }
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = str(Path(args.checkpoint).resolve())
        build_kwargs["load_from_HF"] = False

    print("[INFO] Building SAM3 image model with tracker enabled...")
    model = build_sam3_image_model(**build_kwargs)
    if model.inst_interactive_predictor is None:
        raise RuntimeError("SAM3 image model did not expose inst_interactive_predictor.")
    model.eval()
    return model


def _parse_csv(text: str) -> list[str]:
    values = []
    for item in text.split(","):
        value = item.strip()
        if value and value not in values:
            values.append(value)
    return values


def _build_export_variants(model, precisions: list[str]):
    from src.utils import ExportVariant
    from python.onnx_runtime_policy import DEFAULT_MAX_MEM_FRAMES, MULTI_ANNOTATION_MAX_MEM_FRAMES

    tracker = model.inst_interactive_predictor.model
    native_mem = int(tracker.num_maskmem)
    native_obj_ptrs = int(tracker.max_obj_ptrs_in_encoder)
    variants = []

    for precision in precisions:
        if precision not in ("fp32", "fp16"):
            raise SystemExit("--precisions must contain only fp32 and/or fp16.")
        variants.extend(
            [
                ExportVariant(
                    name="single",
                    precision=precision,
                    max_mem_frames=min(native_mem, DEFAULT_MAX_MEM_FRAMES),
                    max_obj_ptrs=native_obj_ptrs,
                    static_memory_shapes=True,
                ),
                ExportVariant(
                    name="multi",
                    precision=precision,
                    max_mem_frames=min(native_mem, MULTI_ANNOTATION_MAX_MEM_FRAMES),
                    max_obj_ptrs=native_obj_ptrs,
                    static_memory_shapes=True,
                ),
            ]
        )
    return variants


def _save_video_constants(model, outdir: Path, variant) -> None:
    tracker = model.inst_interactive_predictor.model
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 1008, 1008, dtype=torch.float32)
        backbone_out = model.backbone.forward_image(dummy)["sam2_backbone_out"].copy()
        backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
        backbone_out["vision_pos_enc"] = list(backbone_out["vision_pos_enc"])
        backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        _, _feats, pos_embeds, _feat_sizes = tracker._prepare_backbone_features(backbone_out)

    constants_path = outdir / variant.filename("video_constants.npz")
    np.savez(
        constants_path,
        no_mem_embed_bchw=tracker.no_mem_embed.detach().cpu().numpy().reshape(1, 256, 1, 1),
        current_vision_pos_embed=pos_embeds[-1].detach().cpu().numpy(),
        num_maskmem=np.array([tracker.num_maskmem], dtype=np.int64),
        max_obj_ptrs=np.array([tracker.max_obj_ptrs_in_encoder], dtype=np.int64),
        max_cond_frames_in_attn=np.array([tracker.max_cond_frames_in_attn], dtype=np.int64),
        keep_first_cond_frame=np.array([1 if tracker.keep_first_cond_frame else 0], dtype=np.int64),
        memory_temporal_stride_for_eval=np.array(
            [tracker.memory_temporal_stride_for_eval],
            dtype=np.int64,
        ),
        use_memory_selection=np.array([1 if tracker.use_memory_selection else 0], dtype=np.int64),
        mf_threshold=np.array([float(getattr(tracker, "mf_threshold", 0.01))], dtype=np.float32),
        export_max_mem_frames=np.array([int(variant.max_mem_frames or tracker.num_maskmem)], dtype=np.int64),
        export_max_obj_ptrs=np.array([int(variant.max_obj_ptrs or tracker.max_obj_ptrs_in_encoder)], dtype=np.int64),
    )
    print(f"[INFO] Saved video constants [{variant.token or 'default'}]: {constants_path}")


def main(args) -> None:
    sam3_repo = Path(args.sam3_repo).resolve() if args.sam3_repo else _default_sam3_repo()
    if not sam3_repo.exists():
        raise SystemExit(
            f"SAM3 repo not found at {sam3_repo}. Pass --sam3-repo to the local facebookresearch/sam3 clone."
        )

    _add_import_paths(sam3_repo)
    _install_optional_sam3_stubs()

    from src.modules import ImageDecoder, MemAttention, MemEncoder
    from src.utils import (
        export_image_decoder,
        export_memory_attention,
        export_memory_encoder,
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] SAM3 repo   : {sam3_repo}")
    print(f"[INFO] Output dir  : {outdir}")
    if args.checkpoint:
        print(f"[INFO] Checkpoint  : {Path(args.checkpoint).resolve()}")
    else:
        print("[INFO] Checkpoint  : Hugging Face cache/download")

    model = _build_model(args)

    decoder = ImageDecoder(model).eval().cpu()
    mem_attn = MemAttention(model).eval().cpu()
    mem_enc = MemEncoder(model).eval().cpu()

    precisions = _parse_csv(args.precisions)
    export_variants = _build_export_variants(model, precisions)
    if not export_variants:
        raise SystemExit("No export precisions were selected.")

    print(
        "[INFO] Exports     : "
        + ", ".join(variant.token or "default" for variant in export_variants)
    )

    for variant in export_variants:
        export_image_decoder(decoder, str(outdir), variant)
        export_memory_attention(mem_attn, str(outdir), variant)
        export_memory_encoder(mem_enc, str(outdir), variant)
        _save_video_constants(model, outdir, variant)

    print("[INFO] Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export SAM3 tracker-only video modules to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional local SAM3 checkpoint (.pt).",
    )
    parser.add_argument(
        "--load-from-hf",
        action="store_true",
        help="Load the official SAM3 checkpoint from Hugging Face instead of --checkpoint.",
    )
    parser.add_argument(
        "--sam3-repo",
        default="",
        help="Path to a local clone of the SAM3 repo. Defaults to ../sam3 next to this repo.",
    )
    parser.add_argument(
        "--outdir",
        default=str(_repo_root() / "checkpoints" / "sam3" / "video_onnx"),
        help="Directory where the exported ONNX files will be written.",
    )
    parser.add_argument(
        "--precisions",
        default="fp32",
        help="Comma-separated precisions to emit for the internal single/multi tracker bundles: fp32 and/or fp16.",
    )
    main(parser.parse_args())
