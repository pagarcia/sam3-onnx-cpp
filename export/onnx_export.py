# sam3-onnx-cpp/export/onnx_export.py
import argparse
import contextlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"
if str(_PYTHON_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR.resolve()))

from local_sam3_config import DEFAULT_SAM3_REPO


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _export_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_sam3_repo() -> Path:
    return DEFAULT_SAM3_REPO


def _default_outdir_for_version(version: str) -> Path:
    checkpoints_dir = _repo_root() / "checkpoints" / "sam3"
    if version == "sam3.1":
        return checkpoints_dir / "sam31_video_onnx"
    return checkpoints_dir / "video_onnx"


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


def _parse_csv(text: str) -> list[str]:
    values = []
    for item in text.split(","):
        value = item.strip()
        if value and value not in values:
            values.append(value)
    return values


def _validate_precisions(precisions: list[str]) -> list[str]:
    if not precisions:
        raise SystemExit("No export precisions were selected.")
    invalid = [precision for precision in precisions if precision not in ("fp32", "fp16")]
    if invalid:
        raise SystemExit("--precisions must contain only fp32 and/or fp16.")
    return precisions


def _build_sam3_model(args):
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


def _resolve_sam31_checkpoint_path(args):
    from sam3.model_builder import download_ckpt_from_hf

    if args.checkpoint:
        return Path(args.checkpoint).resolve()
    if args.load_from_hf:
        return Path(download_ckpt_from_hf(version="sam3.1")).resolve()
    raise SystemExit(
        "Provide --checkpoint or pass --load-from-hf so the SAM 3.1 exporter does not run on random weights."
    )


def _remap_sam31_tracker_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        new_key = None
        if key.startswith("tracker.model."):
            new_key = key[len("tracker.model.") :]
        elif key.startswith("detector.backbone."):
            new_key = "backbone." + key[len("detector.backbone.") :]
        elif key.startswith("sam2_predictor."):
            new_key = key[len("sam2_predictor.") :]
        elif key.startswith("sam3_model.backbone."):
            new_key = "backbone." + key[len("sam3_model.backbone.") :]
        if new_key:
            remapped[new_key] = value
    if not remapped:
        raise RuntimeError(
            "Could not find tracker.model.* or detector.backbone.* keys in the SAM 3.1 checkpoint."
        )
    return remapped


def _build_sam31_tracker(args):
    from sam3.model_builder import build_sam3_multiplex_video_model

    checkpoint_path = _resolve_sam31_checkpoint_path(args)
    tracker = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        device="cpu",
        use_fa3=False,
        use_rope_real=True,
        strict_state_dict_loading=False,
    )
    tracker.eval()

    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    remapped_state = _remap_sam31_tracker_state_dict(state_dict)
    missing_keys, unexpected_keys = tracker.load_state_dict(remapped_state, strict=False)
    if missing_keys:
        raise RuntimeError(
            "Failed to load the SAM 3.1 tracker export model. Missing keys include "
            f"{missing_keys[:10]}."
        )
    if unexpected_keys:
        print(
            "[WARN] Ignoring detector-only SAM 3.1 checkpoint keys during tracker export: "
            f"{unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
        )
    return tracker, checkpoint_path


def _build_sam3_export_variants(model, precisions: list[str]):
    from src.utils import ExportVariant
    from python.onnx_runtime_policy import DEFAULT_MAX_MEM_FRAMES, MULTI_ANNOTATION_MAX_MEM_FRAMES

    tracker = model.inst_interactive_predictor.model
    native_mem = int(tracker.num_maskmem)
    native_obj_ptrs = int(tracker.max_obj_ptrs_in_encoder)
    variants = []

    for precision in _validate_precisions(precisions):
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


def _build_sam31_export_variants(precisions: list[str]):
    from src.utils import ExportVariant

    return [ExportVariant(name="", precision=precision) for precision in _validate_precisions(precisions)]


def _extract_backbone_pos_embeds(model, tracker, *, use_cpu_bf16_autocast: bool):
    autocast_ctx = contextlib.nullcontext()
    if use_cpu_bf16_autocast:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    with torch.no_grad(), autocast_ctx:
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
    return pos_embeds


def _extract_sam31_backbone_features(tracker, *, use_cpu_bf16_autocast: bool):
    from sam3.model.data_misc import NestedTensor

    autocast_ctx = contextlib.nullcontext()
    if use_cpu_bf16_autocast:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    with torch.no_grad(), autocast_ctx:
        dummy = torch.zeros(1, 3, 1008, 1008, dtype=torch.float32)
        backbone_out = tracker.forward_image(
            NestedTensor(tensors=dummy, mask=None),
            need_interactive_out=True,
            need_propagation_out=True,
        )
        return tracker._prepare_backbone_features(backbone_out)


def _save_video_constants(model, outdir: Path, variant) -> None:
    tracker = model.inst_interactive_predictor.model
    try:
        pos_embeds = _extract_backbone_pos_embeds(
            model,
            tracker,
            use_cpu_bf16_autocast=False,
        )
    except RuntimeError as exc:
        if "must have the same dtype" not in str(exc):
            raise
        print(
            "[WARN] Falling back to CPU bf16 autocast while extracting video constants. "
            "This is needed for newer SAM3/SAM3.1 backbones on Windows.",
        )
        pos_embeds = _extract_backbone_pos_embeds(
            model,
            tracker,
            use_cpu_bf16_autocast=True,
        )

    constants_path = outdir / variant.filename("video_constants.npz")
    np.savez(
        constants_path,
        no_mem_embed_bchw=tracker.no_mem_embed.detach()
        .to(torch.float32)
        .cpu()
        .numpy()
        .reshape(1, 256, 1, 1),
        current_vision_pos_embed=pos_embeds[-1].detach().to(torch.float32).cpu().numpy(),
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


def _save_sam31_constants(tracker, outdir: Path, variant) -> None:
    try:
        backbone_features = _extract_sam31_backbone_features(
            tracker,
            use_cpu_bf16_autocast=False,
        )
    except RuntimeError as exc:
        if "must have the same dtype" not in str(exc):
            raise
        print(
            "[WARN] Falling back to CPU bf16 autocast while extracting SAM 3.1 constants. "
            "This is needed for newer SAM3/SAM3.1 backbones on Windows.",
        )
        backbone_features = _extract_sam31_backbone_features(
            tracker,
            use_cpu_bf16_autocast=True,
        )

    constants = {
        "interactive_vision_pos_embed": backbone_features["interactive"]["vision_pos_embeds"][-1]
        .detach()
        .to(torch.float32)
        .cpu()
        .numpy(),
        "current_vision_pos_embed": backbone_features["sam2_backbone_out"]["vision_pos_embeds"][-1]
        .detach()
        .to(torch.float32)
        .cpu()
        .numpy(),
        "interactive_dense_pe": tracker.interactive_sam_prompt_encoder.get_dense_pe()
        .detach()
        .to(torch.float32)
        .cpu()
        .numpy(),
        "propagation_dense_pe": tracker.get_propagation_dense_pe()
        .detach()
        .to(torch.float32)
        .cpu()
        .numpy(),
        "interactivity_no_mem_embed": tracker.interactivity_no_mem_embed.detach()
        .to(torch.float32)
        .cpu()
        .numpy(),
        "maskmem_tpos_enc": tracker.maskmem_tpos_enc.detach().to(torch.float32).cpu().numpy(),
        "output_valid_embed": tracker.output_valid_embed.detach().to(torch.float32).cpu().numpy(),
        "output_invalid_embed": tracker.output_invalid_embed.detach().to(torch.float32).cpu().numpy(),
        "multiplex_count": np.array([tracker.multiplex_count], dtype=np.int64),
        "num_maskmem": np.array([tracker.num_maskmem], dtype=np.int64),
        "max_obj_ptrs": np.array([tracker.max_obj_ptrs_in_encoder], dtype=np.int64),
        "image_size": np.array([tracker.image_size], dtype=np.int64),
        "sigmoid_scale_for_mem_enc": np.array(
            [float(tracker.sigmoid_scale_for_mem_enc)],
            dtype=np.float32,
        ),
        "sigmoid_bias_for_mem_enc": np.array(
            [float(tracker.sigmoid_bias_for_mem_enc)],
            dtype=np.float32,
        ),
        "object_score_logit_threshold": np.array(
            [float(tracker.object_score_logit_threshold)],
            dtype=np.float32,
        ),
    }
    if tracker.no_obj_embed_spatial is not None:
        constants["no_obj_embed_spatial"] = (
            tracker.no_obj_embed_spatial.detach().to(torch.float32).cpu().numpy()
        )

    constants_path = outdir / variant.filename("sam31_constants.npz")
    np.savez(constants_path, **constants)
    print(f"[INFO] Saved SAM 3.1 constants [{variant.token or 'default'}]: {constants_path}")


def _save_sam31_manifest(
    outdir: Path,
    checkpoint_path: Path,
    variant,
) -> None:
    manifest = {
        "version": "sam3.1",
        "checkpoint": str(checkpoint_path),
        "precision": variant.precision,
        "files": {
            "interactive_decoder": variant.filename("interactive_decoder.onnx"),
            "propagation_decoder": variant.filename("propagation_decoder.onnx"),
            "memory_encoder": variant.filename("memory_encoder.onnx"),
            "memory_attention_core": variant.filename("memory_attention_core.onnx"),
            "constants": variant.filename("sam31_constants.npz"),
        },
        "notes": [
            "This bundle exports SAM 3.1 tracker components only.",
            "The image encoder ONNX export remains disabled in this repo.",
            "memory_attention_core.onnx expects token-space inputs with any object-pointer slots already padded into memory_image_tokens and memory_image_pos.",
        ],
    }
    manifest_path = outdir / variant.filename("sam31_export_info.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Saved SAM 3.1 manifest [{variant.token or 'default'}]: {manifest_path}")


def main(args) -> None:
    sam3_repo = Path(args.sam3_repo).resolve() if args.sam3_repo else _default_sam3_repo()
    if not sam3_repo.exists():
        raise SystemExit(
            f"SAM3 repo not found at {sam3_repo}. Pass --sam3-repo or set SAM3_REPO to the local facebookresearch/sam3 clone."
        )

    _add_import_paths(sam3_repo)
    _install_optional_sam3_stubs()

    outdir = Path(args.outdir).resolve() if args.outdir else _default_outdir_for_version(args.version)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] SAM3 repo   : {sam3_repo}")
    print(f"[INFO] Version     : {args.version}")
    print(f"[INFO] Output dir  : {outdir}")
    if args.checkpoint:
        print(f"[INFO] Checkpoint  : {Path(args.checkpoint).resolve()}")
    else:
        print("[INFO] Checkpoint  : Hugging Face cache/download")

    precisions = _parse_csv(args.precisions)

    if args.version == "sam3":
        from src.modules import ImageDecoder, MemAttention, MemEncoder
        from src.utils import (
            export_image_decoder,
            export_memory_attention,
            export_memory_encoder,
        )

        model = _build_sam3_model(args)
        decoder = ImageDecoder(model).eval().cpu()
        mem_attn = MemAttention(model).eval().cpu()
        mem_enc = MemEncoder(model).eval().cpu()
        export_variants = _build_sam3_export_variants(model, precisions)

        print(
            "[INFO] Exports     : "
            + ", ".join(variant.token or "default" for variant in export_variants)
        )

        for variant in export_variants:
            export_image_decoder(decoder, str(outdir), variant)
            export_memory_attention(mem_attn, str(outdir), variant)
            export_memory_encoder(mem_enc, str(outdir), variant)
            _save_video_constants(model, outdir, variant)
    else:
        from src.sam31_modules import (
            SAM31InteractiveDecoder,
            SAM31MemoryAttentionCore,
            SAM31MemoryEncoder,
            SAM31PropagationDecoder,
        )
        from src.utils import (
            export_sam31_interactive_decoder,
            export_sam31_memory_attention_core,
            export_sam31_memory_encoder,
            export_sam31_propagation_decoder,
        )

        tracker, checkpoint_path = _build_sam31_tracker(args)
        interactive_decoder = SAM31InteractiveDecoder(tracker).eval().cpu()
        propagation_decoder = SAM31PropagationDecoder(tracker).eval().cpu()
        memory_encoder = SAM31MemoryEncoder(tracker).eval().cpu()
        memory_attention_core = SAM31MemoryAttentionCore(tracker).eval().cpu()
        export_variants = _build_sam31_export_variants(precisions)

        print(
            "[INFO] Exports     : "
            + ", ".join(variant.token or "default" for variant in export_variants)
        )

        for variant in export_variants:
            export_sam31_interactive_decoder(interactive_decoder, str(outdir), variant)
            export_sam31_propagation_decoder(propagation_decoder, str(outdir), variant)
            export_sam31_memory_encoder(memory_encoder, str(outdir), variant)
            export_sam31_memory_attention_core(memory_attention_core, str(outdir), variant)
            _save_sam31_constants(tracker, outdir, variant)
            _save_sam31_manifest(outdir, checkpoint_path, variant)

    print("[INFO] Export complete.")


if __name__ == "__main__":
    _configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Export SAM3 or SAM 3.1 video modules to ONNX."
    )
    parser.add_argument(
        "--version",
        default="sam3",
        choices=("sam3", "sam3.1"),
        help="Model family to export. Use sam3.1 for the multiplex tracker-only bundle.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional local checkpoint (.pt). For --version sam3.1 this should be the merged sam3.1_multiplex.pt checkpoint.",
    )
    parser.add_argument(
        "--load-from-hf",
        action="store_true",
        help="Load the official checkpoint from Hugging Face instead of --checkpoint.",
    )
    parser.add_argument(
        "--sam3-repo",
        default="",
        help="Path to a local clone of the SAM3 repo. Defaults to SAM3_REPO or an auto-detected sibling checkout such as ../sam3-3p1 or ../sam3.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Directory where the exported ONNX files will be written. Defaults to checkpoints/sam3/video_onnx for sam3 and checkpoints/sam3/sam31_video_onnx for sam3.1.",
    )
    parser.add_argument(
        "--precisions",
        default="fp32,fp16",
        help="Comma-separated precisions to emit. sam3 keeps single/multi tracker variants; sam3.1 emits one bundle per selected precision.",
    )
    main(parser.parse_args())
