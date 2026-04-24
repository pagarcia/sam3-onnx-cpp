#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from onnx_test_utils import (
    empty_boxes,
    empty_points,
    prepare_boxes,
    prepare_points,
    preprocess_image_bgr,
    run_decoder,
    run_encoder,
    set_cv2_threads,
)
from quantize_image_models import (
    default_onnx_dir,
    external_data_path,
    quantize_model,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    encoder_input: Path
    encoder_output: Path | None
    encoder_ops: list[str] | None
    encoder_preprocess: bool
    decoder_input: Path
    decoder_output: Path | None
    decoder_ops: list[str] | None
    decoder_preprocess: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_output_dir() -> Path:
    return repo_root() / "checkpoints" / "sam3" / "benchmarks" / "image_cpu"


def resolve_image_path(arg_value: str) -> Path:
    if arg_value:
        image_path = Path(arg_value).expanduser().resolve()
        if not image_path.exists():
            raise SystemExit(f"Image file does not exist: {image_path}")
        return image_path

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select an Image",
        "",
        "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)",
    )
    if not img_path:
        raise SystemExit("No image selected - exiting.")
    return Path(img_path).expanduser().resolve()


def parse_threads_spec(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise SystemExit("--threads produced no usable values.")

    resolved: list[int] = []
    cpu_count = os.cpu_count() or 8
    for part in parts:
        if part == "auto":
            resolved.append(max(1, cpu_count - 1))
        else:
            try:
                resolved.append(max(1, int(part)))
            except ValueError as exc:
                raise SystemExit(f"Invalid thread count {part!r}") from exc
    deduped: list[int] = []
    seen = set()
    for value_int in resolved:
        if value_int not in seen:
            deduped.append(value_int)
            seen.add(value_int)
    return deduped


def parse_points_spec(text: str) -> tuple[list[tuple[int, int]], list[int]]:
    points: list[tuple[int, int]] = []
    labels: list[int] = []
    if not text.strip():
        return points, labels

    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"Invalid point spec {item!r}; expected x,y,label")
        x, y, label = int(parts[0]), int(parts[1]), int(parts[2])
        points.append((x, y))
        labels.append(label)
    return points, labels


def parse_box_spec(text: str) -> tuple[int, int, int, int] | None:
    if not text.strip():
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--box must be x1,y1,x2,y2")
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def build_prompt_inputs(info,
                        image_shape: tuple[int, int],
                        point_text: str,
                        box_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    box = parse_box_spec(box_text)
    points, labels = parse_points_spec(point_text)

    if box is not None:
        boxes = prepare_boxes(box, info)
        pts, lbl = empty_points()
        return pts, lbl, boxes, "box"

    if not points:
        height, width = image_shape
        points = [(width // 2, height // 2)]
        labels = [1]
        mode = "default_center_point"
    else:
        mode = "points"

    pts, lbl = prepare_points(points, labels, info)
    return pts, lbl, empty_boxes(), mode


def make_cpu_session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = int(os.getenv("SAM3_ORT_INTER_OP_THREADS", "1"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = os.getenv("SAM3_ORT_CPU_ARENA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )
    so.enable_mem_pattern = os.getenv("SAM3_ORT_MEM_PATTERN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def ensure_variant_artifact(spec: VariantSpec,
                            cache_dir: Path,
                            force_quantize: bool) -> tuple[Path, Path, dict[str, object]]:
    metadata: dict[str, object] = {}

    encoder_path = spec.encoder_input
    if spec.encoder_output is not None:
        encoder_path = cache_dir / spec.encoder_output
        if force_quantize or not encoder_path.exists():
            metadata["encoder_quant"] = quantize_model(
                spec.encoder_input,
                encoder_path,
                "encoder",
                do_preprocess=spec.encoder_preprocess,
                op_types=spec.encoder_ops or ["MatMul"],
            )

    decoder_path = spec.decoder_input
    if spec.decoder_output is not None:
        decoder_path = cache_dir / spec.decoder_output
        if force_quantize or not decoder_path.exists():
            metadata["decoder_quant"] = quantize_model(
                spec.decoder_input,
                decoder_path,
                "decoder",
                do_preprocess=spec.decoder_preprocess,
                op_types=spec.decoder_ops or ["MatMul", "Gemm"],
            )

    return encoder_path, decoder_path, metadata


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def median_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def variant_specs(onnx_dir: Path) -> list[VariantSpec]:
    return [
        VariantSpec(
            name="fp32_fp32",
            encoder_input=onnx_dir / "vision_encoder.onnx",
            encoder_output=None,
            encoder_ops=None,
            encoder_preprocess=False,
            decoder_input=onnx_dir / "prompt_encoder_mask_decoder.onnx",
            decoder_output=None,
            decoder_ops=None,
            decoder_preprocess=False,
        ),
        VariantSpec(
            name="enc_int8_matmul__dec_fp32",
            encoder_input=onnx_dir / "vision_encoder.onnx",
            encoder_output=Path("vision_encoder.int8.matmul.onnx"),
            encoder_ops=["MatMul"],
            encoder_preprocess=False,
            decoder_input=onnx_dir / "prompt_encoder_mask_decoder.onnx",
            decoder_output=None,
            decoder_ops=None,
            decoder_preprocess=False,
        ),
        VariantSpec(
            name="enc_int8_matmul_gather__dec_fp32",
            encoder_input=onnx_dir / "vision_encoder.onnx",
            encoder_output=Path("vision_encoder.int8.matmul_gather.onnx"),
            encoder_ops=["MatMul", "Gather"],
            encoder_preprocess=False,
            decoder_input=onnx_dir / "prompt_encoder_mask_decoder.onnx",
            decoder_output=None,
            decoder_ops=None,
            decoder_preprocess=False,
        ),
        VariantSpec(
            name="enc_int8_matmul_gather_pre__dec_fp32",
            encoder_input=onnx_dir / "vision_encoder.onnx",
            encoder_output=Path("vision_encoder.int8.matmul_gather_pre.onnx"),
            encoder_ops=["MatMul", "Gather"],
            encoder_preprocess=True,
            decoder_input=onnx_dir / "prompt_encoder_mask_decoder.onnx",
            decoder_output=None,
            decoder_ops=None,
            decoder_preprocess=False,
        ),
        VariantSpec(
            name="enc_int8_matmul_gather__dec_int8",
            encoder_input=onnx_dir / "vision_encoder.onnx",
            encoder_output=Path("vision_encoder.int8.matmul_gather.onnx"),
            encoder_ops=["MatMul", "Gather"],
            encoder_preprocess=False,
            decoder_input=onnx_dir / "prompt_encoder_mask_decoder.onnx",
            decoder_output=Path("prompt_encoder_mask_decoder.int8.matmul_gemm.onnx"),
            decoder_ops=["MatMul", "Gemm"],
            decoder_preprocess=False,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 image CPU variants across encoder/decoder quantization choices."
    )
    parser.add_argument("--image", default="", help="Optional image path. If omitted, a file picker is shown.")
    parser.add_argument(
        "--threads",
        default="auto",
        help="Comma-separated CPU thread counts, for example 6,7,8. 'auto' means cpu_count-1.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed runs per variant/thread pair. First run is reported separately from steady-state.",
    )
    parser.add_argument(
        "--points",
        default="",
        help="Optional point prompt as x,y,label;x,y,label. Defaults to a center positive point.",
    )
    parser.add_argument("--box", default="", help="Optional box prompt as x1,y1,x2,y2.")
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant filter. Defaults to all built-in variants.",
    )
    parser.add_argument(
        "--force_quantize",
        action="store_true",
        help="Regenerate benchmark quantized artifacts even if cached copies exist.",
    )
    parser.add_argument(
        "--onnx_dir",
        default=str(default_onnx_dir()),
        help="Directory containing the SAM3 image ONNX artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(default_output_dir()),
        help="Directory for benchmark JSON/CSV summaries.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive.")

    image_path = resolve_image_path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    set_cv2_threads(1)
    onnx_dir = Path(args.onnx_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = onnx_dir / "bench_cpu"
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_threads = parse_threads_spec(args.threads)
    selected_variants = variant_specs(onnx_dir)
    if args.variants.strip():
        wanted = {part.strip() for part in args.variants.split(",") if part.strip()}
        selected_variants = [variant for variant in selected_variants if variant.name in wanted]
        if not selected_variants:
            raise SystemExit("No variants matched --variants.")

    print(f"[INFO] Image: {image_path}")
    print(f"[INFO] Threads: {selected_threads}")
    print(f"[INFO] Repeats: {args.repeats}")
    print(f"[INFO] Variants: {[variant.name for variant in selected_variants]}")

    results: list[dict[str, object]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant in selected_variants:
        print(f"[INFO] Preparing variant {variant.name} ...")
        encoder_path, decoder_path, metadata = ensure_variant_artifact(
            variant,
            cache_dir=cache_dir,
            force_quantize=args.force_quantize,
        )

        for threads in selected_threads:
            print(f"[INFO] Benchmarking {variant.name} @ threads={threads} ...")

            session_start = time.perf_counter()
            sess_enc = make_cpu_session(encoder_path, threads=threads)
            sess_dec = make_cpu_session(decoder_path, threads=threads)
            session_load_ms = (time.perf_counter() - session_start) * 1000.0

            run_rows: list[dict[str, float]] = []
            prompt_mode_used = ""
            for _ in range(args.repeats):
                prep_start = time.perf_counter()
                pixel_values, info = preprocess_image_bgr(image_bgr, target_size=1008)
                prep_ms = (time.perf_counter() - prep_start) * 1000.0

                points, labels, boxes, prompt_mode_used = build_prompt_inputs(
                    info,
                    image_shape=image_bgr.shape[:2],
                    point_text=args.points,
                    box_text=args.box,
                )

                enc_start = time.perf_counter()
                enc_out = run_encoder(sess_enc, pixel_values)
                enc_ms = (time.perf_counter() - enc_start) * 1000.0

                dec_start = time.perf_counter()
                _ = run_decoder(
                    sess_dec,
                    enc_out,
                    input_points=points,
                    input_labels=labels,
                    input_boxes=boxes,
                )
                dec_ms = (time.perf_counter() - dec_start) * 1000.0

                run_rows.append({
                    "prep_ms": prep_ms,
                    "enc_ms": enc_ms,
                    "dec_ms": dec_ms,
                    "total_ms": prep_ms + enc_ms + dec_ms,
                })

            first = run_rows[0]
            steady = run_rows[1:] if len(run_rows) > 1 else run_rows

            row = {
                "variant": variant.name,
                "threads": threads,
                "encoder_path": str(encoder_path),
                "decoder_path": str(decoder_path),
                "encoder_size_bytes": encoder_path.stat().st_size if encoder_path.exists() else 0,
                "decoder_size_bytes": decoder_path.stat().st_size if decoder_path.exists() else 0,
                "session_load_ms": session_load_ms,
                "prompt_mode": prompt_mode_used,
                "repeat_count": len(run_rows),
                "prep_first_ms": first["prep_ms"],
                "enc_first_ms": first["enc_ms"],
                "dec_first_ms": first["dec_ms"],
                "total_first_ms": first["total_ms"],
                "prep_steady_mean_ms": mean_or_zero(item["prep_ms"] for item in steady),
                "enc_steady_mean_ms": mean_or_zero(item["enc_ms"] for item in steady),
                "dec_steady_mean_ms": mean_or_zero(item["dec_ms"] for item in steady),
                "total_steady_mean_ms": mean_or_zero(item["total_ms"] for item in steady),
                "total_steady_median_ms": median_or_zero(item["total_ms"] for item in steady),
            }
            if "encoder_quant" in metadata:
                row["encoder_preprocess_used"] = metadata["encoder_quant"].get("preprocess_used")
                row["encoder_ops"] = ",".join(metadata["encoder_quant"].get("op_types", []))
            if "decoder_quant" in metadata:
                row["decoder_preprocess_used"] = metadata["decoder_quant"].get("preprocess_used")
                row["decoder_ops"] = ",".join(metadata["decoder_quant"].get("op_types", []))

            results.append(row)

            print(
                "[INFO] "
                f"{variant.name} threads={threads} "
                f"first_total={row['total_first_ms']:.1f} ms "
                f"steady_total_mean={row['total_steady_mean_ms']:.1f} ms "
                f"enc_steady_mean={row['enc_steady_mean_ms']:.1f} ms "
                f"dec_steady_mean={row['dec_steady_mean_ms']:.1f} ms"
            )

    results.sort(key=lambda item: (float(item["total_steady_mean_ms"]), float(item["enc_steady_mean_ms"])))

    json_path = output_dir / f"benchmark_image_cpu_variants_{timestamp}.json"
    csv_path = output_dir / f"benchmark_image_cpu_variants_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "image": str(image_path),
                "threads": selected_threads,
                "repeats": args.repeats,
                "variants": [variant.name for variant in selected_variants],
                "results": results,
            },
            handle,
            indent=2,
        )

    if results:
        fieldnames: list[str] = []
        seen_fields: set[str] = set()
        for row in results:
            for key in row.keys():
                if key in seen_fields:
                    continue
                seen_fields.add(key)
                fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"[INFO] JSON summary: {json_path}")
    print(f"[INFO] CSV summary : {csv_path}")
    if results:
        best = results[0]
        print(
            "[INFO] Best steady-state config: "
            f"{best['variant']} @ threads={best['threads']} "
            f"=> total={best['total_steady_mean_ms']:.1f} ms "
            f"(enc={best['enc_steady_mean_ms']:.1f} ms, dec={best['dec_steady_mean_ms']:.1f} ms)"
        )


if __name__ == "__main__":
    main()
