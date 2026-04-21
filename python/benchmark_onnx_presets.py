#!/usr/bin/env python3
import argparse
import csv
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
from sweep_onnx_mem_frames import _aggregate_summary


PRESETS = {
    "fast": {"max_mem_frames": 2, "max_obj_ptrs": 16},
    "quality": {"max_mem_frames": 7, "max_obj_ptrs": 16},
    "parity": {"max_mem_frames": 7, "max_obj_ptrs": 16},
}


def _write_csv(path: Path, rows) -> None:
    fieldnames = [
        "preset",
        "onnx_variant",
        "onnx_preset_requested",
        "onnx_max_mem_frames",
        "onnx_max_obj_ptrs",
        "repeat_count",
        "repeat_mean_iou_median",
        "repeat_mean_iou_std",
        "repeat_min_iou_median",
        "repeat_min_iou_std",
        "onnx_repeat_median_mean_total_ms",
        "onnx_repeat_median_mean_steady_total_ms",
        "onnx_repeat_std_mean_total_ms",
        "speedup_vs_native_repeat_median_mean",
        "speedup_vs_native_repeat_median_mean_steady",
        "native_repeat_median_mean_total_ms",
        "native_repeat_median_mean_steady_total_ms",
        "onnx_mean_prep_ms",
        "onnx_mean_attn_ms",
        "onnx_median_attn_ms",
        "onnx_mean_enc_ms",
        "onnx_median_enc_ms",
        "onnx_mean_dec_ms",
        "onnx_median_dec_ms",
        "onnx_mean_mem_ms",
        "onnx_median_mem_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the recommended ONNX presets against the same native SAM3 baseline."
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
        "--presets",
        nargs="+",
        default=["fast", "quality", "parity"],
        choices=sorted(PRESETS.keys()),
        help="Named ONNX presets to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated native and ONNX runs per preset.",
    )
    parser.add_argument(
        "--onnx_accel",
        default="cuda",
        choices=["auto", "cpu", "cuda", "trt"],
        help="Execution provider choice for the ONNX subprocess.",
    )
    parser.add_argument(
        "--onnx_variant",
        default="",
        help="Optional ONNX tracker variant suffix such as fp16, fast, quality, or quality_fp16.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory for the preset benchmark outputs.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations in the ONNX subprocess.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    outdir = Path(args.outdir).resolve() if args.outdir else Path(
        tempfile.mkdtemp(prefix="sam3_preset_bench_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
    )
    outdir.mkdir(parents=True, exist_ok=True)

    video = _load_video_frames(args.video, args.max_frames)
    prompt_spec = _resolve_prompt(args, video.raw_frames[0])
    prompt_json_path = Path(args.save_prompt_json).resolve() if args.save_prompt_json else outdir / "prompt.json"
    _save_prompt_spec(prompt_json_path, prompt_spec)

    summary_json = outdir / "preset_summary.json"
    summary_csv = outdir / "preset_summary.csv"
    native_dir = outdir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Presets: {args.presets}")
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
    for preset_name in args.presets:
        preset = PRESETS[preset_name]
        run_dir = outdir / preset_name
        run_dir.mkdir(parents=True, exist_ok=True)
        onnx_runs = []

        for repeat_idx in range(1, args.repeats + 1):
            onnx_npz = run_dir / f"repeat_{repeat_idx:02d}.npz"
            print(
                f"[INFO] Running preset={preset_name} "
                f"(mem={preset['max_mem_frames']}, obj_ptrs={preset['max_obj_ptrs']}) "
                f"repeat {repeat_idx}/{args.repeats}..."
            )
            _run_onnx_subprocess(
                video_path=args.video,
                onnx_dir=Path(args.onnx_dir).resolve(),
                prompt_json=prompt_json_path,
                save_path=onnx_npz,
                max_frames=len(video.raw_frames),
                safe=args.safe,
                onnx_accel=args.onnx_accel,
                onnx_max_mem_frames=preset["max_mem_frames"],
                onnx_max_obj_ptrs=preset["max_obj_ptrs"],
                onnx_variant=args.onnx_variant,
                onnx_preset=preset_name,
            )
            onnx_runs.append(_load_npz(onnx_npz))

        summary = _aggregate_summary(
            native_runs=native_runs,
            onnx_runs=onnx_runs,
            mem_frames=preset["max_mem_frames"],
            onnx_max_obj_ptrs=preset["max_obj_ptrs"],
        )
        summary["preset"] = preset_name
        summary["video"] = str(args.video)
        summary["prompt"] = prompt_spec

        preset_summary_json = run_dir / "summary.json"
        with preset_summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        rows.append(summary)

        print(
            f"[INFO] preset={preset_name} | "
            f"IoU={summary['repeat_mean_iou_median']:.4f} | "
            f"repeat-median steady ONNX="
            f"{summary.get('onnx_repeat_median_mean_steady_total_ms', summary['onnx_repeat_median_mean_total_ms']):.1f} ms/frame | "
            f"steady speedup="
            f"{summary.get('speedup_vs_native_repeat_median_mean_steady', summary['speedup_vs_native_repeat_median_mean']):.2f}x"
        )

    fastest = (
        min(
            rows,
            key=lambda item: item.get(
                "onnx_repeat_median_mean_steady_total_ms",
                item["onnx_repeat_median_mean_total_ms"],
            ),
        )
        if rows
        else None
    )
    best_iou = max(rows, key=lambda item: item["repeat_mean_iou_median"]) if rows else None

    payload = {
        "video": str(args.video),
        "frame_count": len(video.raw_frames),
        "prompt": prompt_spec,
        "presets": {name: PRESETS[name] for name in args.presets},
        "onnx_accel": args.onnx_accel,
        "onnx_variant": args.onnx_variant,
        "repeats": int(args.repeats),
        "fastest": fastest,
        "best_iou": best_iou,
        "results": rows,
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_csv(summary_csv, rows)

    if fastest is not None:
        print(
            f"[INFO] Fastest preset: {fastest['preset']} "
            f"at {fastest.get('onnx_repeat_median_mean_steady_total_ms', fastest['onnx_repeat_median_mean_total_ms']):.1f} ms/frame "
            f"({fastest.get('speedup_vs_native_repeat_median_mean_steady', fastest['speedup_vs_native_repeat_median_mean']):.2f}x native)"
        )
    if best_iou is not None:
        print(
            f"[INFO] Best IoU preset: {best_iou['preset']} "
            f"with repeat-median mean IoU {best_iou['repeat_mean_iou_median']:.4f}"
        )
    print(f"[INFO] Summary JSON : {summary_json}")
    print(f"[INFO] Summary CSV  : {summary_csv}")
    print(f"[INFO] Native dir   : {native_dir}")


if __name__ == "__main__":
    main()
