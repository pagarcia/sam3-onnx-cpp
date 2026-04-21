#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtWidgets

from compare_native_vs_onnx import (
    DEFAULT_CKPT,
    DEFAULT_ONNX_DIR,
    DEFAULT_SAM3_REPO,
    REPO_ROOT,
    _build_native_model,
    _compute_display_base,
    _compute_native_backbone_features,
    _mask_to_uint8,
    _measure_torch,
    _prepare_native_point_inputs,
    _preprocess_frame_native,
    _release_cuda_memory,
    _resolve_prompt,
    _save_prompt_spec,
)


RESULT_LAYOUT = (
    ("Native", "native"),
    ("ONNX Fast", "fast"),
    ("ONNX Quality", "quality"),
    ("ONNX Parity", "parity"),
)


def _resolve_image_path(arg_value: str) -> str:
    if arg_value:
        image_path = Path(arg_value).expanduser().resolve()
        if not image_path.exists():
            raise SystemExit(f"Image file does not exist: {image_path}")
        return str(image_path)

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)

    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select an Image",
        "",
        "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All files (*.*)",
    )
    if owns_app:
        app.quit()
    if not img_path:
        raise SystemExit("No image selected.")
    return img_path


def _prompt_is_empty(prompt_spec: dict) -> bool:
    if prompt_spec["prompt"] == "bounding_box":
        return prompt_spec.get("box") is None
    return not prompt_spec.get("points")


def _green_overlay(image_bgr: np.ndarray, mask_uint8: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = image_bgr.copy()
    fg = mask_uint8 > 0
    color = np.zeros_like(image_bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(overlay, 1.0, color, alpha, 0)


def _draw_prompt(vis_bgr: np.ndarray, prompt_spec: dict, scale: float) -> None:
    if prompt_spec["prompt"] == "bounding_box":
        box = prompt_spec.get("box")
        if box is None:
            return
        x1, y1, x2, y2 = [int(round(value * scale)) for value in box]
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return

    for item in prompt_spec.get("points", []):
        px = int(round(item[0] * scale))
        py = int(round(item[1] * scale))
        label = int(item[2])
        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        cv2.circle(vis_bgr, (px, py), 6, color, -1)


def _annotate_tile(vis_bgr: np.ndarray, title: str, meta: dict) -> np.ndarray:
    tile = vis_bgr.copy()
    header_h = 72
    cv2.rectangle(tile, (0, 0), (tile.shape[1], header_h), (18, 18, 18), -1)
    cv2.putText(tile, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    line1 = f"Total {meta['full_total_ms']:.1f} ms"
    line2 = (
        f"Prep {meta['prep_ms']:.1f} | Enc {meta['enc_ms']:.1f} | "
        f"Attn {meta['attn_ms']:.1f}"
    )
    line3 = f"Dec {meta['dec_ms']:.1f} | Mem {meta['mem_ms']:.1f}"
    cv2.putText(tile, line1, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, line2, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(tile, line3, (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 215, 215), 1, cv2.LINE_AA)

    if meta.get("variant"):
        cv2.putText(
            tile,
            f"Variant: {meta['variant']}",
            (tile.shape[1] - 260, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 235, 180),
            1,
            cv2.LINE_AA,
        )
    return tile


def _compose_grid(image_bgr: np.ndarray, prompt_spec: dict, results: dict[str, dict]) -> np.ndarray:
    tile_base, scale = _compute_display_base(image_bgr, max_side=840)
    tiles: list[np.ndarray] = []

    for title, key in RESULT_LAYOUT:
        meta = results[key]
        mask = cv2.resize(
            meta["mask_uint8"],
            (tile_base.shape[1], tile_base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        vis = _green_overlay(tile_base, mask)
        _draw_prompt(vis, prompt_spec, scale)
        tiles.append(_annotate_tile(vis, title, meta))

    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:])
    return np.vstack([top, bottom])


def _run_native_image_compare(
    image_bgr: np.ndarray,
    prompt_spec: dict,
    checkpoint: str,
    sam3_repo: Path,
) -> tuple[np.ndarray, dict]:
    image_model = _build_native_model(sam3_repo, checkpoint)
    tracker = image_model.inst_interactive_predictor.model
    device = next(image_model.parameters()).device
    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    try:
        point_inputs = _prepare_native_point_inputs(
            prompt_spec,
            image_bgr.shape[1],
            image_bgr.shape[0],
            device,
        )
        prep_t0 = time.time()
        frame_cpu = _preprocess_frame_native(image_bgr)
        prep_ms = (time.time() - prep_t0) * 1000.0

        frame_t0 = time.time()
        image = frame_cpu.to(device, non_blocking=True).unsqueeze(0)
        (_, current_vision_feats, current_vision_pos_embeds, feat_sizes), enc_ms = _measure_torch(
            lambda: _compute_native_backbone_features(image_model, tracker, image)
        )

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *shape)
                for x, shape in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        fused_embed, attn_ms = _measure_torch(
            lambda: tracker._prepare_memory_conditioned_features(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=1,
                track_in_reverse=False,
                use_prev_mem_frame=True,
            )
        )

        multimask_output = tracker._use_multimask(True, point_inputs)
        sam_outputs, dec_ms = _measure_torch(
            lambda: tracker._forward_sam_heads(
                backbone_features=fused_embed,
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        )
        (
            _,
            _high_res_multimasks,
            _ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        mem_out, mem_ms = _measure_torch(
            lambda: tracker._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=True,
                output_dict=output_dict,
                is_init_cond_frame=True,
            )
        )
        maskmem_features, maskmem_pos_enc = mem_out
        output_dict["cond_frame_outputs"][0] = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": low_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        mask_uint8 = _mask_to_uint8(high_res_masks, image_bgr.shape[1], image_bgr.shape[0])
        full_total_ms = ((time.time() - frame_t0) * 1000.0) + prep_ms
        meta = {
            "prep_ms": float(prep_ms),
            "enc_ms": float(enc_ms),
            "attn_ms": float(attn_ms),
            "dec_ms": float(dec_ms),
            "mem_ms": float(mem_ms),
            "full_total_ms": float(full_total_ms),
            "variant": "native",
        }
        return mask_uint8, meta
    finally:
        del output_dict, tracker, image_model
        _release_cuda_memory()


def _run_onnx_worker(
    *,
    preset: str,
    image_path: str,
    prompt_json_path: Path,
    onnx_dir: Path,
    outdir: Path,
    safe: bool,
    onnx_variant: str,
) -> tuple[np.ndarray, dict]:
    mask_path = outdir / f"{preset}_mask.png"
    json_path = outdir / f"{preset}_summary.json"
    cmd = [
        str(REPO_ROOT / "sam3_env" / "Scripts" / "python.exe"),
        str(REPO_ROOT / "python" / "onnx_compare_image_worker.py"),
        "--image",
        str(Path(image_path).resolve()),
        "--prompt_json",
        str(prompt_json_path.resolve()),
        "--preset",
        preset,
        "--onnx_dir",
        str(onnx_dir.resolve()),
        "--save_mask",
        str(mask_path),
        "--save_json",
        str(json_path),
    ]
    if safe:
        cmd.append("--safe")
    if onnx_variant:
        cmd.extend(["--onnx_variant", onnx_variant])

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    mask_uint8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_uint8 is None:
        raise RuntimeError(f"Failed to read ONNX mask: {mask_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    meta = {
        "prep_ms": float(payload["prep_ms"]),
        "enc_ms": float(payload["enc_ms"]),
        "attn_ms": float(payload["attn_ms"]),
        "dec_ms": float(payload["dec_ms"]),
        "mem_ms": float(payload["mem_ms"]),
        "full_total_ms": float(payload["full_total_ms"]),
        "variant": payload["runtime"].get("resolved_variant", preset),
    }
    return mask_uint8, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one native SAM3 image prompt against ONNX fast/quality/parity in a 2x2 grid."
    )
    parser.add_argument("--image", default="", help="Optional image path. If omitted, a file picker is shown.")
    parser.add_argument(
        "--prompt",
        default="seed_points",
        choices=["seed_points", "bounding_box"],
        help="Prompt mode for interactive selection.",
    )
    parser.add_argument("--points", default="", help="Prompt points as x,y,label;x,y,label")
    parser.add_argument("--box", default="", help="Prompt box as x1,y1,x2,y2")
    parser.add_argument("--prompt_json", default="", help="Optional prompt JSON to replay.")
    parser.add_argument("--save_prompt_json", default="", help="Optional output path for the prompt JSON.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to the SAM3 checkpoint for native inference.",
    )
    parser.add_argument(
        "--sam3_repo",
        default=str(DEFAULT_SAM3_REPO),
        help="Path to the local SAM3 repo.",
    )
    parser.add_argument(
        "--onnx_dir",
        default=str(DEFAULT_ONNX_DIR),
        help="Directory containing the exported ONNX tracker files.",
    )
    parser.add_argument(
        "--onnx_variant",
        default="",
        help="Optional explicit ONNX variant override for all presets.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory for prompt/masks/summary/composite.",
    )
    parser.add_argument(
        "--save_compare_png",
        default="",
        help="Optional explicit PNG path for the final 2x2 comparison image.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations in the ONNX workers.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve() if args.outdir else Path(
        tempfile.mkdtemp(prefix="sam3_image_compare_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
    )
    outdir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_image_path(args.image)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    prompt_spec = _resolve_prompt(args, image_bgr)
    if _prompt_is_empty(prompt_spec):
        raise SystemExit("Prompt selection was empty.")

    prompt_json_path = (
        Path(args.save_prompt_json).resolve()
        if args.save_prompt_json
        else outdir / "prompt.json"
    )
    _save_prompt_spec(prompt_json_path, prompt_spec)

    print(f"[INFO] Image     : {image_path}")
    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Prompt    : {prompt_json_path}")

    results: dict[str, dict] = {}
    for preset in ("fast", "quality", "parity"):
        print(f"[INFO] Running ONNX preset: {preset}")
        mask_uint8, meta = _run_onnx_worker(
            preset=preset,
            image_path=image_path,
            prompt_json_path=prompt_json_path,
            onnx_dir=Path(args.onnx_dir),
            outdir=outdir,
            safe=args.safe,
            onnx_variant=args.onnx_variant,
        )
        results[preset] = {
            "mask_uint8": mask_uint8,
            **meta,
        }

    print("[INFO] Running native image path...")
    native_mask, native_meta = _run_native_image_compare(
        image_bgr,
        prompt_spec,
        checkpoint=str(Path(args.checkpoint).resolve()),
        sam3_repo=Path(args.sam3_repo).resolve(),
    )
    results["native"] = {
        "mask_uint8": native_mask,
        **native_meta,
    }

    grid = _compose_grid(image_bgr, prompt_spec, results)
    compare_png = (
        Path(args.save_compare_png).resolve()
        if args.save_compare_png
        else outdir / "compare_2x2.png"
    )
    compare_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(compare_png), grid):
        raise RuntimeError(f"Failed to save comparison image: {compare_png}")

    summary = {
        "image": str(Path(image_path).resolve()),
        "prompt_json": str(prompt_json_path),
        "native": {key: value for key, value in results["native"].items() if key != "mask_uint8"},
        "fast": {key: value for key, value in results["fast"].items() if key != "mask_uint8"},
        "quality": {key: value for key, value in results["quality"].items() if key != "mask_uint8"},
        "parity": {key: value for key, value in results["parity"].items() if key != "mask_uint8"},
        "compare_png": str(compare_png),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Saved comparison image: {compare_png}")
    print(f"[INFO] Saved summary JSON    : {summary_path}")
    cv2.namedWindow("SAM3 2D Compare", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("SAM3 2D Compare", grid)
    print("[INFO] Showing 2x2 comparison. Press ESC to close.")
    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
