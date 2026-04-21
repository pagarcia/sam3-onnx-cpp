#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from sam3_onnx_session import Sam3OnnxTrackerSession, load_prompt_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the default ONNX image prompt path and save the mask/timings."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt_json", required=True, help="Prompt JSON path.")
    parser.add_argument(
        "--onnx_dir",
        default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "sam3" / "video_onnx"),
        help="Directory containing the exported ONNX tracker files.",
    )
    parser.add_argument("--save_mask", required=True, help="Output mask PNG path.")
    parser.add_argument("--save_json", required=True, help="Output JSON summary path.")
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations for all sessions.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    prompt_spec = load_prompt_spec(Path(args.prompt_json).resolve())
    tracker = Sam3OnnxTrackerSession(
        Path(args.onnx_dir),
        safe=args.safe,
    )
    prepared, prompt_points, prompt_labels = tracker.prepare_prompt_from_spec(
        image_bgr,
        prompt_spec,
    )
    result = tracker.process_frame(
        0,
        image_bgr,
        prepared=prepared,
        prompt_points=prompt_points,
        prompt_labels=prompt_labels,
    )

    save_mask = Path(args.save_mask).resolve()
    save_mask.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_mask), result.mask_uint8):
        raise SystemExit(f"Failed to write mask image: {save_mask}")

    save_json = Path(args.save_json).resolve()
    save_json.parent.mkdir(parents=True, exist_ok=True)
    full_total_ms = (
        float(result.timings.prep_ms)
        + float(result.timings.enc_ms)
        + float(result.timings.total_ms)
    )
    payload = {
        "image": str(image_path),
        "mode": "default",
        "mask_path": str(save_mask),
        "prep_ms": float(result.timings.prep_ms),
        "enc_ms": float(result.timings.enc_ms),
        "attn_ms": float(result.timings.attn_ms),
        "dec_ms": float(result.timings.dec_ms),
        "mem_ms": float(result.timings.mem_ms),
        "stage_total_ms": float(result.timings.total_ms),
        "full_total_ms": full_total_ms,
        "runtime": tracker.runtime_metadata,
    }
    save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] mode=default full_total_ms={full_total_ms:.1f}")
    print(f"[INFO] Saved mask: {save_mask}")
    print(f"[INFO] Saved JSON: {save_json}")


if __name__ == "__main__":
    main()
