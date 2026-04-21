#!/usr/bin/env python3
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from compare_native_vs_onnx import (
    DEFAULT_CKPT,
    DEFAULT_ONNX_DIR,
    DEFAULT_SAM3_REPO,
    REPO_ROOT,
    _decode_json_scalar,
    _frame_metrics,
    _load_npz,
    _load_video_frames,
    _resolve_prompt,
    _run_native_tracker,
    _run_onnx_subprocess,
    _save_prompt_spec,
)


TIMING_KEYS = ("prep_ms", "enc_ms", "attn_ms", "dec_ms", "mem_ms", "total_ms")


def _parse_mem_frames(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--mem_frames values must be positive integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--mem_frames produced an empty sweep")
    return values


def _summarize_timing_series(prefix: str, key: str, arrays: list[np.ndarray], summary: dict) -> None:
    non_empty = [arr for arr in arrays if arr.size > 0]
    if not non_empty:
        return

    stacked = np.concatenate(non_empty, axis=0)
    repeat_means = np.asarray([arr.mean() for arr in non_empty], dtype=np.float32)
    repeat_medians = np.asarray([np.median(arr) for arr in non_empty], dtype=np.float32)

    summary[f"{prefix}_mean_{key}"] = float(stacked.mean())
    summary[f"{prefix}_median_{key}"] = float(np.median(stacked))
    summary[f"{prefix}_p90_{key}"] = float(np.percentile(stacked, 90))
    summary[f"{prefix}_repeat_mean_{key}"] = float(repeat_means.mean())
    summary[f"{prefix}_repeat_median_mean_{key}"] = float(np.median(repeat_means))
    summary[f"{prefix}_repeat_std_mean_{key}"] = float(repeat_means.std())
    summary[f"{prefix}_repeat_median_{key}"] = float(repeat_medians.mean())
    summary[f"{prefix}_repeat_median_median_{key}"] = float(np.median(repeat_medians))
    summary[f"{prefix}_repeat_std_median_{key}"] = float(repeat_medians.std())


def _aggregate_timing_arrays(prefix: str, runs: list[dict]) -> dict:
    summary = {
        f"{prefix}_repeat_count": int(len(runs)),
    }
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    for key in TIMING_KEYS:
        arrays = [np.asarray(run[key], dtype=np.float32) for run in runs if key in run]
        arrays_by_key[key] = arrays
        _summarize_timing_series(prefix, key, arrays, summary)

    total_arrays = arrays_by_key.get("total_ms", [])
    _summarize_timing_series(
        prefix,
        "frame0_total_ms",
        [arr[:1] for arr in total_arrays if arr.size > 0],
        summary,
    )
    _summarize_timing_series(
        prefix,
        "steady_total_ms",
        [arr[1:] if arr.size > 1 else arr for arr in total_arrays if arr.size > 0],
        summary,
    )

    mean_total = summary.get(f"{prefix}_mean_total_ms")
    median_total = summary.get(f"{prefix}_median_total_ms")
    repeat_median_mean_total = summary.get(f"{prefix}_repeat_median_mean_total_ms")
    mean_steady_total = summary.get(f"{prefix}_mean_steady_total_ms")
    repeat_median_mean_steady_total = summary.get(f"{prefix}_repeat_median_mean_steady_total_ms")

    summary[f"{prefix}_fps"] = float(1000.0 / mean_total) if mean_total and mean_total > 0 else 0.0
    summary[f"{prefix}_median_fps"] = (
        float(1000.0 / median_total) if median_total and median_total > 0 else 0.0
    )
    summary[f"{prefix}_repeat_median_mean_fps"] = (
        float(1000.0 / repeat_median_mean_total)
        if repeat_median_mean_total and repeat_median_mean_total > 0
        else 0.0
    )
    summary[f"{prefix}_steady_fps"] = (
        float(1000.0 / mean_steady_total) if mean_steady_total and mean_steady_total > 0 else 0.0
    )
    summary[f"{prefix}_repeat_median_mean_steady_fps"] = (
        float(1000.0 / repeat_median_mean_steady_total)
        if repeat_median_mean_steady_total and repeat_median_mean_steady_total > 0
        else 0.0
    )
    return summary


def _aggregate_quality(native_ref: dict, onnx_runs: list[dict]) -> dict:
    repeat_summaries = []
    all_frame_summaries = []

    native_frame_count = len(native_ref["masks"])
    onnx_frame_count = min(len(run["masks"]) for run in onnx_runs)
    frame_count = min(native_frame_count, onnx_frame_count)

    for run_idx, onnx in enumerate(onnx_runs, start=1):
        frame_summaries = [
            _frame_metrics(native_ref["masks"][frame_idx], onnx["masks"][frame_idx])
            for frame_idx in range(frame_count)
        ]
        all_frame_summaries.extend(frame_summaries)

        if frame_summaries:
            repeat_mean_iou = float(np.mean([item["iou"] for item in frame_summaries]))
            repeat_min_iou = float(np.min([item["iou"] for item in frame_summaries]))
            repeat_mean_dice = float(np.mean([item["dice"] for item in frame_summaries]))
            repeat_mean_pixel_acc = float(np.mean([item["pixel_acc"] for item in frame_summaries]))
        else:
            repeat_mean_iou = 0.0
            repeat_min_iou = 0.0
            repeat_mean_dice = 0.0
            repeat_mean_pixel_acc = 0.0

        repeat_summaries.append(
            {
                "repeat_idx": run_idx,
                "mean_iou": repeat_mean_iou,
                "min_iou": repeat_min_iou,
                "mean_dice": repeat_mean_dice,
                "mean_pixel_acc": repeat_mean_pixel_acc,
            }
        )

    if all_frame_summaries:
        mean_iou = float(np.mean([item["iou"] for item in all_frame_summaries]))
        min_iou = float(np.min([item["iou"] for item in all_frame_summaries]))
        mean_dice = float(np.mean([item["dice"] for item in all_frame_summaries]))
        mean_pixel_acc = float(np.mean([item["pixel_acc"] for item in all_frame_summaries]))
        worst_global_idx = int(np.argmin([item["iou"] for item in all_frame_summaries]))
        worst_frame_idx = worst_global_idx % frame_count
        worst_frame = {
            "repeat_idx": int(worst_global_idx // frame_count) + 1,
            "frame_idx": worst_frame_idx,
            **all_frame_summaries[worst_global_idx],
        }
    else:
        mean_iou = 0.0
        min_iou = 0.0
        mean_dice = 0.0
        mean_pixel_acc = 0.0
        worst_frame = None

    repeat_mean_ious = np.asarray([item["mean_iou"] for item in repeat_summaries], dtype=np.float32)
    repeat_min_ious = np.asarray([item["min_iou"] for item in repeat_summaries], dtype=np.float32)
    repeat_mean_dices = np.asarray([item["mean_dice"] for item in repeat_summaries], dtype=np.float32)
    repeat_mean_pixel_accs = np.asarray(
        [item["mean_pixel_acc"] for item in repeat_summaries], dtype=np.float32
    )

    return {
        "frame_count": int(frame_count),
        "native_frame_count": int(native_frame_count),
        "onnx_frame_count": int(onnx_frame_count),
        "repeat_count": int(len(onnx_runs)),
        "mean_iou": mean_iou,
        "min_iou": min_iou,
        "mean_dice": mean_dice,
        "mean_pixel_acc": mean_pixel_acc,
        "repeat_mean_iou_median": float(np.median(repeat_mean_ious)),
        "repeat_mean_iou_std": float(repeat_mean_ious.std()),
        "repeat_min_iou_median": float(np.median(repeat_min_ious)),
        "repeat_min_iou_std": float(repeat_min_ious.std()),
        "repeat_mean_dice_median": float(np.median(repeat_mean_dices)),
        "repeat_mean_dice_std": float(repeat_mean_dices.std()),
        "repeat_mean_pixel_acc_median": float(np.median(repeat_mean_pixel_accs)),
        "repeat_mean_pixel_acc_std": float(repeat_mean_pixel_accs.std()),
        "worst_frame": worst_frame,
        "per_repeat": repeat_summaries,
    }


def _aggregate_summary(native_runs: list[dict], onnx_runs: list[dict], mem_frames: int, onnx_max_obj_ptrs: int):
    quality = _aggregate_quality(native_runs[0], onnx_runs)
    summary = {
        "onnx_max_mem_frames": int(mem_frames),
        "onnx_max_obj_ptrs": int(onnx_max_obj_ptrs),
    }
    summary.update(quality)
    summary.update(_aggregate_timing_arrays("native", native_runs))
    summary.update(_aggregate_timing_arrays("onnx", onnx_runs))
    onnx_runtime = _decode_json_scalar(onnx_runs[0], "runtime_json") if onnx_runs else None
    if isinstance(onnx_runtime, dict):
        summary["onnx_variant"] = onnx_runtime.get("resolved_variant", "")
        summary["onnx_preset_requested"] = onnx_runtime.get("requested_preset", "")
        summary["onnx_runtime"] = onnx_runtime

    native_mean_total = summary["native_mean_total_ms"]
    onnx_mean_total = summary["onnx_mean_total_ms"]
    native_median_total = summary["native_median_total_ms"]
    onnx_median_total = summary["onnx_median_total_ms"]
    native_repeat_median_mean_total = summary["native_repeat_median_mean_total_ms"]
    onnx_repeat_median_mean_total = summary["onnx_repeat_median_mean_total_ms"]
    native_mean_steady_total = summary.get("native_mean_steady_total_ms", native_mean_total)
    onnx_mean_steady_total = summary.get("onnx_mean_steady_total_ms", onnx_mean_total)
    native_repeat_median_mean_steady_total = summary.get(
        "native_repeat_median_mean_steady_total_ms",
        native_repeat_median_mean_total,
    )
    onnx_repeat_median_mean_steady_total = summary.get(
        "onnx_repeat_median_mean_steady_total_ms",
        onnx_repeat_median_mean_total,
    )

    summary["speedup_vs_native_mean"] = float(native_mean_total / onnx_mean_total) if onnx_mean_total > 0 else 0.0
    summary["speedup_vs_native_median"] = (
        float(native_median_total / onnx_median_total) if onnx_median_total > 0 else 0.0
    )
    summary["speedup_vs_native_repeat_median_mean"] = (
        float(native_repeat_median_mean_total / onnx_repeat_median_mean_total)
        if onnx_repeat_median_mean_total > 0
        else 0.0
    )
    summary["speedup_vs_native_mean_steady"] = (
        float(native_mean_steady_total / onnx_mean_steady_total) if onnx_mean_steady_total > 0 else 0.0
    )
    summary["speedup_vs_native_repeat_median_mean_steady"] = (
        float(native_repeat_median_mean_steady_total / onnx_repeat_median_mean_steady_total)
        if onnx_repeat_median_mean_steady_total > 0
        else 0.0
    )
    summary["onnx_total_delta_mean_ms"] = float(onnx_mean_total - native_mean_total)
    summary["onnx_total_delta_median_ms"] = float(onnx_median_total - native_median_total)
    summary["onnx_steady_total_delta_mean_ms"] = float(onnx_mean_steady_total - native_mean_steady_total)
    return summary


def _write_csv(path: Path, rows) -> None:
    fieldnames = [
        "onnx_max_mem_frames",
        "onnx_variant",
        "onnx_preset_requested",
        "repeat_count",
        "mean_iou",
        "repeat_mean_iou_median",
        "repeat_mean_iou_std",
        "min_iou",
        "repeat_min_iou_median",
        "repeat_min_iou_std",
        "mean_dice",
        "repeat_mean_dice_median",
        "repeat_mean_dice_std",
        "native_mean_total_ms",
        "native_median_total_ms",
        "native_mean_frame0_total_ms",
        "native_mean_steady_total_ms",
        "native_repeat_median_mean_total_ms",
        "native_repeat_median_mean_steady_total_ms",
        "native_repeat_std_mean_total_ms",
        "onnx_mean_total_ms",
        "onnx_median_total_ms",
        "onnx_mean_frame0_total_ms",
        "onnx_mean_steady_total_ms",
        "onnx_repeat_median_mean_total_ms",
        "onnx_repeat_median_mean_steady_total_ms",
        "onnx_repeat_std_mean_total_ms",
        "speedup_vs_native_mean",
        "speedup_vs_native_median",
        "speedup_vs_native_repeat_median_mean",
        "speedup_vs_native_mean_steady",
        "speedup_vs_native_repeat_median_mean_steady",
        "native_mean_prep_ms",
        "onnx_mean_prep_ms",
        "native_mean_attn_ms",
        "native_median_attn_ms",
        "onnx_mean_attn_ms",
        "onnx_median_attn_ms",
        "native_mean_enc_ms",
        "native_median_enc_ms",
        "onnx_mean_enc_ms",
        "onnx_median_enc_ms",
        "native_mean_dec_ms",
        "native_median_dec_ms",
        "onnx_mean_dec_ms",
        "onnx_median_dec_ms",
        "native_mean_mem_ms",
        "native_median_mem_ms",
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
        description="Sweep ONNX spatial memory-frame caps against repeated native SAM3 baselines."
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
        "--mem_frames",
        default="2,3,4,5,6,7",
        help="Comma-separated ONNX spatial memory-frame caps to sweep.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated native and ONNX runs per memory-frame setting.",
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
        "--onnx_max_obj_ptrs",
        type=int,
        default=0,
        help="Optional cap on ONNX object pointers. 0 keeps the exported tracker default.",
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

    mem_frames_values = _parse_mem_frames(args.mem_frames)
    outdir = Path(args.outdir).resolve() if args.outdir else Path(
        tempfile.mkdtemp(prefix="sam3_mem_sweep_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
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
    print(f"[INFO] ONNX mem sweep: {mem_frames_values}")
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
    for mem_frames in mem_frames_values:
        run_dir = outdir / f"mem_{mem_frames}"
        run_dir.mkdir(parents=True, exist_ok=True)
        onnx_runs = []

        for repeat_idx in range(1, args.repeats + 1):
            onnx_npz = run_dir / f"repeat_{repeat_idx:02d}.npz"
            print(
                f"[INFO] Running ONNX tracker with max_mem_frames={mem_frames} "
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
                onnx_max_mem_frames=mem_frames,
                onnx_max_obj_ptrs=args.onnx_max_obj_ptrs,
                onnx_variant=args.onnx_variant,
            )
            onnx_runs.append(_load_npz(onnx_npz))

        summary = _aggregate_summary(native_runs, onnx_runs, mem_frames, args.onnx_max_obj_ptrs)
        summary["video"] = str(args.video)
        summary["prompt"] = prompt_spec

        summary_json = run_dir / "summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        rows.append(summary)

        print(
            f"[INFO] mem={mem_frames} | IoU={summary['mean_iou']:.4f} | "
            f"steady ONNX={summary.get('onnx_repeat_median_mean_steady_total_ms', summary['onnx_repeat_median_mean_total_ms']):.1f} ms/frame | "
            f"steady native={summary.get('native_repeat_median_mean_steady_total_ms', summary['native_repeat_median_mean_total_ms']):.1f} ms/frame | "
            f"steady speedup={summary.get('speedup_vs_native_repeat_median_mean_steady', summary['speedup_vs_native_repeat_median_mean']):.2f}x"
        )

    ranking = sorted(
        rows,
        key=lambda item: (
            item.get("onnx_repeat_median_mean_steady_total_ms", item["onnx_repeat_median_mean_total_ms"]),
            -item["repeat_mean_iou_median"],
        ),
    )
    best_stable_speed = ranking[0] if ranking else None
    best_iou = max(rows, key=lambda item: item["repeat_mean_iou_median"]) if rows else None

    sweep_payload = {
        "video": str(args.video),
        "frame_count": len(video.raw_frames),
        "prompt": prompt_spec,
        "mem_frames_values": mem_frames_values,
        "onnx_accel": args.onnx_accel,
        "onnx_variant": args.onnx_variant,
        "repeats": int(args.repeats),
        "onnx_max_obj_ptrs": int(args.onnx_max_obj_ptrs),
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
            f"[INFO] Best stable speed: mem={best_stable_speed['onnx_max_mem_frames']} "
            f"at repeat-median steady mean "
            f"{best_stable_speed.get('onnx_repeat_median_mean_steady_total_ms', best_stable_speed['onnx_repeat_median_mean_total_ms']):.1f} ms/frame "
            f"({best_stable_speed.get('speedup_vs_native_repeat_median_mean_steady', best_stable_speed['speedup_vs_native_repeat_median_mean']):.2f}x native)"
        )
    if best_iou is not None:
        print(
            f"[INFO] Best IoU setting: mem={best_iou['onnx_max_mem_frames']} "
            f"with repeat-median mean IoU {best_iou['repeat_mean_iou_median']:.4f}"
        )
    print(f"[INFO] Sweep JSON : {sweep_json}")
    print(f"[INFO] Sweep CSV  : {sweep_csv}")
    print(f"[INFO] Native dir : {native_dir}")


if __name__ == "__main__":
    main()
