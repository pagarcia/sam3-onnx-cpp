#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

from compare_native_vs_onnx import (
    DEFAULT_CKPT,
    DEFAULT_ONNX_DIR,
    DEFAULT_SAM3_REPO,
    REPO_ROOT,
    _load_npz,
    _load_video_frames,
    _resolve_prompt,
    _run_native_tracker,
    _run_onnx_subprocess,
    _save_prompt_spec,
)
from sweep_onnx_mem_frames import _aggregate_summary, _write_csv


def _parse_obj_ptrs(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--obj_ptrs values must be positive integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--obj_ptrs produced an empty sweep")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Sweep ONNX object-pointer caps against repeated native SAM3 baselines."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--onnx_dir",
        default=str(DEFAULT_ONNX_DIR),
        help="Directory containing the exported ONNX tracker files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to the SAM3 checkpoint.",
    )
    parser.add_argument(
        "--sam3_repo",
        default=str(DEFAULT_SAM3_REPO),
        help="Path to the local SAM3 repo.",
    )
    parser.add_argument(
        "--prompt",
        default="seed_points",
        choices=["seed_points", "bounding_box"],
    )
    parser.add_argument("--points", default="", help="Prompt points as x,y,label;x,y,label")
    parser.add_argument("--box", default="", help="Prompt box as x1,y1,x2,y2")
    parser.add_argument("--prompt_json", default="", help="Optional prompt JSON to replay.")
    parser.add_argument("--save_prompt_json", default="", help="Optional output path for the prompt JSON.")
    parser.add_argument("--max_frames", type=int, default=20, help="Number of frames to compare.")
    parser.add_argument(
        "--max_mem_frames",
        type=int,
        default=7,
        help="Fixed ONNX spatial memory cap to use while sweeping object pointers.",
    )
    parser.add_argument(
        "--obj_ptrs",
        default="2,4,8,12,16",
        help="Comma-separated ONNX object-pointer caps to sweep.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated native and ONNX runs per object-pointer setting.",
    )
    parser.add_argument(
        "--onnx_accel",
        default="cuda",
        choices=["auto", "cpu", "cuda", "trt"],
        help="Execution provider choice for the ONNX subprocess.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory for the sweep outputs.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations in the ONNX subprocess.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.max_mem_frames <= 0:
        raise SystemExit("--max_mem_frames must be positive")

    obj_ptr_values = _parse_obj_ptrs(args.obj_ptrs)
    outdir = Path(args.outdir).resolve() if args.outdir else Path(
        tempfile.mkdtemp(prefix="sam3_objptr_sweep_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
    )
    outdir.mkdir(parents=True, exist_ok=True)

    video = _load_video_frames(args.video, args.max_frames)
    prompt_spec = _resolve_prompt(args, video.raw_frames[0])
    prompt_json_path = Path(args.save_prompt_json).resolve() if args.save_prompt_json else outdir / "prompt.json"
    _save_prompt_spec(prompt_json_path, prompt_spec)

    sweep_json = outdir / "sweep_summary.json"
    sweep_csv = outdir / "sweep_summary.csv"
    native_dir = outdir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Fixed max_mem_frames: {args.max_mem_frames}")
    print(f"[INFO] ONNX obj_ptr sweep: {obj_ptr_values}")
    print(f"[INFO] Repeats: {args.repeats}")

    native_runs = []
    for repeat_idx in range(1, args.repeats + 1):
        native_npz = native_dir / f"repeat_{repeat_idx:02d}.npz"
        print(f"[INFO] Running native PyTorch tracker repeat {repeat_idx}/{args.repeats}...")
        _run_native_tracker(
            video=video,
            prompt_spec=prompt_spec,
            checkpoint=str(Path(args.checkpoint).resolve()),
            sam3_repo=Path(args.sam3_repo).resolve(),
            save_path=native_npz,
            video_path=args.video,
        )
        native_runs.append(_load_npz(native_npz))

    rows = []
    for obj_ptrs in obj_ptr_values:
        run_dir = outdir / f"objptr_{obj_ptrs}"
        run_dir.mkdir(parents=True, exist_ok=True)
        onnx_runs = []

        for repeat_idx in range(1, args.repeats + 1):
            onnx_npz = run_dir / f"repeat_{repeat_idx:02d}.npz"
            print(
                f"[INFO] Running ONNX tracker with max_obj_ptrs={obj_ptrs} "
                f"(repeat {repeat_idx}/{args.repeats})..."
            )
            _run_onnx_subprocess(
                video_path=args.video,
                onnx_dir=Path(args.onnx_dir).resolve(),
                prompt_json=prompt_json_path,
                save_path=onnx_npz,
                max_frames=len(video.raw_frames),
                safe=args.safe,
                onnx_accel=args.onnx_accel,
                onnx_max_mem_frames=args.max_mem_frames,
                onnx_max_obj_ptrs=obj_ptrs,
            )
            onnx_runs.append(_load_npz(onnx_npz))

        summary = _aggregate_summary(native_runs, onnx_runs, args.max_mem_frames, obj_ptrs)
        summary["video"] = str(args.video)
        summary["prompt"] = prompt_spec

        summary_json = run_dir / "summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        rows.append(summary)

        print(
            f"[INFO] obj_ptrs={obj_ptrs} | IoU={summary['mean_iou']:.4f} | "
            f"median ONNX={summary['onnx_median_total_ms']:.1f} ms/frame | "
            f"median native={summary['native_median_total_ms']:.1f} ms/frame | "
            f"repeat-median speedup={summary['speedup_vs_native_repeat_median_mean']:.2f}x"
        )

    ranking = sorted(
        rows,
        key=lambda item: (
            item["onnx_repeat_median_mean_total_ms"],
            -item["repeat_mean_iou_median"],
        ),
    )
    best_stable_speed = ranking[0] if ranking else None
    best_iou = max(rows, key=lambda item: item["repeat_mean_iou_median"]) if rows else None

    sweep_payload = {
        "video": str(args.video),
        "frame_count": len(video.raw_frames),
        "prompt": prompt_spec,
        "fixed_max_mem_frames": int(args.max_mem_frames),
        "obj_ptr_values": obj_ptr_values,
        "onnx_accel": args.onnx_accel,
        "repeats": int(args.repeats),
        "native_defaults": {
            "num_maskmem": 7,
            "memory_temporal_stride_for_eval": 1,
            "max_obj_ptrs_in_encoder": 16,
        },
        "best_stable_speed": best_stable_speed,
        "best_iou": best_iou,
        "results": rows,
    }

    with sweep_json.open("w", encoding="utf-8") as f:
        json.dump(sweep_payload, f, indent=2)
    _write_csv(sweep_csv, rows)

    if best_stable_speed is not None:
        print(
            f"[INFO] Best stable speed: obj_ptrs={best_stable_speed['onnx_max_obj_ptrs']} "
            f"at repeat-median mean {best_stable_speed['onnx_repeat_median_mean_total_ms']:.1f} ms/frame "
            f"({best_stable_speed['speedup_vs_native_repeat_median_mean']:.2f}x native)"
        )
    if best_iou is not None:
        print(
            f"[INFO] Best IoU setting: obj_ptrs={best_iou['onnx_max_obj_ptrs']} "
            f"with repeat-median mean IoU {best_iou['repeat_mean_iou_median']:.4f}"
        )
    print(f"[INFO] Sweep JSON : {sweep_json}")
    print(f"[INFO] Sweep CSV  : {sweep_csv}")
    print(f"[INFO] Native dir : {native_dir}")


if __name__ == "__main__":
    main()
