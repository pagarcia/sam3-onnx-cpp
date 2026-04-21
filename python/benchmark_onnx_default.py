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
from onnx_runtime_policy import resolve_runtime_caps
from sweep_onnx_mem_frames import _aggregate_summary


def _write_csv(path: Path, row: dict) -> None:
    fieldnames = [
        "onnx_mode",
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
        "onnx_mean_enc_ms",
        "onnx_mean_dec_ms",
        "onnx_mean_mem_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the default ONNX SAM3 path against the same native SAM3 baseline."
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
    parser.add_argument(
        "--prompt_json",
        default="",
        help="Optional prompt JSON to replay. Supports the legacy single-frame format or a multi-frame 'annotations' list.",
    )
    parser.add_argument("--save_prompt_json", default="", help="Optional output path for the prompt JSON.")
    parser.add_argument("--max_frames", type=int, default=20, help="Number of frames to compare.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated native and ONNX runs.",
    )
    parser.add_argument(
        "--onnx_accel",
        default="cuda",
        choices=["auto", "cpu", "cuda", "trt"],
        help="Execution provider choice for the ONNX subprocess.",
    )
    parser.add_argument(
        "--onnx_max_mem_frames",
        type=int,
        default=None,
        help="Optional ONNX spatial-memory cap. When omitted, the runtime auto-selects 2 for single-annotation and 4 for multi-annotation.",
    )
    parser.add_argument(
        "--onnx_max_obj_ptrs",
        type=int,
        default=None,
        help="Optional ONNX object-pointer cap. Defaults to 16 when omitted.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory for the benchmark outputs.",
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
        tempfile.mkdtemp(prefix="sam3_default_bench_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
    )
    outdir.mkdir(parents=True, exist_ok=True)

    video = _load_video_frames(args.video, args.max_frames)
    prompt_spec = _resolve_prompt(args, video.raw_frames[0])
    prompt_json_path = Path(args.save_prompt_json).resolve() if args.save_prompt_json else outdir / "prompt.json"
    _save_prompt_spec(prompt_json_path, prompt_spec)
    onnx_max_mem_frames, onnx_max_obj_ptrs = resolve_runtime_caps(
        prompt_spec=prompt_spec,
        max_mem_frames=args.onnx_max_mem_frames,
        max_obj_ptrs=args.onnx_max_obj_ptrs,
    )

    summary_json = outdir / "benchmark_summary.json"
    summary_csv = outdir / "benchmark_summary.csv"
    native_dir = outdir / "native"
    onnx_dir = outdir / "onnx"
    native_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Repeats: {args.repeats}")
    print(
        f"[INFO] Default ONNX runtime caps: max_mem_frames={onnx_max_mem_frames}, "
        f"max_obj_ptrs={onnx_max_obj_ptrs}"
    )

    native_runs = []
    onnx_runs = []
    for repeat_idx in range(1, args.repeats + 1):
        native_npz = native_dir / f"repeat_{repeat_idx:02d}.npz"
        onnx_npz = onnx_dir / f"repeat_{repeat_idx:02d}.npz"

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

        print(f"[INFO] Running default ONNX tracker repeat {repeat_idx}/{args.repeats}...")
        _run_onnx_subprocess(
            video_path=args.video,
            onnx_dir=Path(args.onnx_dir).resolve(),
            prompt_json=prompt_json_path,
            save_path=onnx_npz,
            max_frames=len(video.raw_frames),
            safe=args.safe,
            onnx_accel=args.onnx_accel,
            onnx_max_mem_frames=onnx_max_mem_frames,
            onnx_max_obj_ptrs=onnx_max_obj_ptrs,
        )
        onnx_runs.append(_load_npz(onnx_npz))

    summary = _aggregate_summary(native_runs, onnx_runs, onnx_max_mem_frames, onnx_max_obj_ptrs)
    summary["video"] = str(args.video)
    summary["prompt"] = prompt_spec

    payload = {
        "video": str(args.video),
        "frame_count": len(video.raw_frames),
        "prompt": prompt_spec,
        "repeats": int(args.repeats),
        "onnx_accel": args.onnx_accel,
        "summary": summary,
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_csv(summary_csv, summary)

    print(
        f"[INFO] Mean IoU: {summary['repeat_mean_iou_median']:.4f} | "
        f"steady ONNX={summary.get('onnx_repeat_median_mean_steady_total_ms', summary['onnx_repeat_median_mean_total_ms']):.1f} ms/frame | "
        f"steady speedup={summary.get('speedup_vs_native_repeat_median_mean_steady', summary['speedup_vs_native_repeat_median_mean']):.2f}x"
    )
    print(f"[INFO] Summary JSON : {summary_json}")
    print(f"[INFO] Summary CSV  : {summary_csv}")
    print(f"[INFO] Native dir   : {native_dir}")
    print(f"[INFO] ONNX dir     : {onnx_dir}")


if __name__ == "__main__":
    main()
