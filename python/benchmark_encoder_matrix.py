#!/usr/bin/env python3
"""Benchmark the SAM3 vision encoder across precision x provider x graph-opt.

Answers, headlessly and in minutes, the questions that otherwise need full
Grow3D GUI runs:
  - is fp16 actually faster than fp32 on this machine's DML/CPU?
  - how much does ONNX Runtime graph optimization buy on the encoder?
  - do optimized outputs stay numerically consistent with the fp32 baseline?

Example:
  python benchmark_encoder_matrix.py --providers dml,cpu --runs 5
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

OPT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

VARIANT_FILES = {
    "fp32": "vision_encoder.onnx",
    "fp16": "vision_encoder_fp16.onnx",
    "int8": "vision_encoder.int8.onnx",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def make_input(size: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((1, 3, size, size), dtype=np.float32)


def build_session(model_path: Path, provider: str, opt_name: str, threads: int,
                  mimic_cpp_dml: bool) -> tuple[ort.InferenceSession, float]:
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVELS[opt_name]
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if provider == "dml":
        # Memory pattern must be off for the DML EP (ORT requirement).
        so.enable_mem_pattern = False
        providers = ["DmlExecutionProvider"]
        if mimic_cpp_dml:
            # Mirrors the current C++ runtime DML settings for a true baseline.
            so.add_session_config_entry("session.disable_prepacking", "1")
            so.add_session_config_entry("session.disable_gemm_fast_gelu_fusion", "1")
    else:
        so.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    load_s = time.perf_counter() - t0
    return session, load_s


def primary_output(outputs: list[np.ndarray]) -> np.ndarray:
    # image_embeddings.2 / current vision feature is the last, smallest map;
    # use the largest output for a stricter comparison instead.
    return max(outputs, key=lambda o: o.size)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", default=str(repo_root() / "checkpoints" / "sam3" / "onnx"))
    parser.add_argument("--providers", default="dml,cpu")
    parser.add_argument("--variants", default="fp32,fp16,int8")
    parser.add_argument("--opt-levels", default="disable,basic,extended,all")
    parser.add_argument("--cpu-opt-levels", default="disable,all",
                        help="Reduced level set for slow CPU configs.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cpu-runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--size", type=int, default=1008)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    opt_levels = [o.strip() for o in args.opt_levels.split(",") if o.strip()]
    cpu_opt_levels = [o.strip() for o in args.cpu_opt_levels.split(",") if o.strip()]

    available = ort.get_available_providers()
    if "dml" in providers and "DmlExecutionProvider" not in available:
        print("[WARN] DmlExecutionProvider not available; dropping dml configs.")
        providers = [p for p in providers if p != "dml"]

    input_tensor = make_input(args.size)
    results: list[dict] = []

    reference: np.ndarray | None = None
    fp32_path = onnx_dir / VARIANT_FILES["fp32"]
    if fp32_path.exists():
        print("[INFO] computing cpu/fp32 reference output...")
        ref_session, _ = build_session(fp32_path, "cpu", "disable", args.threads, False)
        ref_name = ref_session.get_inputs()[0].name
        reference = primary_output(ref_session.run(None, {ref_name: input_tensor})).astype(np.float32)
        del ref_session
        gc.collect()

    for provider in providers:
        levels = cpu_opt_levels if provider == "cpu" else opt_levels
        runs = args.cpu_runs if provider == "cpu" else args.runs
        for variant in variants:
            if provider == "dml" and variant == "int8":
                # Dynamic-int8 ops mostly fall back to CPU under DML; skip.
                continue
            model_path = onnx_dir / VARIANT_FILES[variant]
            if not model_path.exists():
                print(f"[WARN] missing {model_path.name}; skipping {provider}/{variant}")
                continue
            for opt_name in levels:
                mimic = provider == "dml" and opt_name == "disable"
                tag = f"{provider:>3} {variant:>4} opt={opt_name:<8}"
                try:
                    session, load_s = build_session(
                        model_path, provider, opt_name, args.threads, mimic)
                    input_name = session.get_inputs()[0].name
                    t0 = time.perf_counter()
                    outputs = session.run(None, {input_name: input_tensor})
                    first_s = time.perf_counter() - t0
                    for _ in range(max(0, args.warmup - 1)):
                        session.run(None, {input_name: input_tensor})
                    times = []
                    for _ in range(runs):
                        t0 = time.perf_counter()
                        outputs = session.run(None, {input_name: input_tensor})
                        times.append(time.perf_counter() - t0)
                    out = primary_output(outputs).astype(np.float32)
                    if reference is None and provider == "cpu" and variant == "fp32":
                        reference = out.copy()
                    cos = cosine(out, reference) if reference is not None else float("nan")
                    row = {
                        "provider": provider,
                        "variant": variant,
                        "opt": opt_name,
                        "load_s": round(load_s, 2),
                        "first_run_s": round(first_s, 2),
                        "mean_ms": round(1000.0 * float(np.mean(times)), 1),
                        "min_ms": round(1000.0 * float(np.min(times)), 1),
                        "cos_vs_cpu_fp32": round(cos, 6) if cos == cos else None,
                    }
                    results.append(row)
                    print(f"[RESULT] {tag} load={row['load_s']:7.2f}s "
                          f"first={row['first_run_s']:7.2f}s mean={row['mean_ms']:8.1f}ms "
                          f"min={row['min_ms']:8.1f}ms cos={row['cos_vs_cpu_fp32']}")
                    del session, outputs
                    gc.collect()
                except Exception as exc:  # noqa: BLE001 - report and continue the matrix
                    print(f"[ERROR] {tag} failed: {exc}")
                    results.append({
                        "provider": provider, "variant": variant, "opt": opt_name,
                        "error": str(exc),
                    })

    # cpu fp32 baseline may have been produced after other rows; recompute cosines
    if reference is not None:
        print("\n[NOTE] cosine column compares the largest encoder output against "
              "the cpu/fp32 run from this same matrix (fixed random input).")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[INFO] wrote {args.json_out}")

    print("\n| provider | variant | opt | load s | first s | mean ms | min ms | cos vs cpu-fp32 |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for row in results:
        if "error" in row:
            print(f"| {row['provider']} | {row['variant']} | {row['opt']} | ERROR: {row['error'][:60]} |")
            continue
        print(f"| {row['provider']} | {row['variant']} | {row['opt']} | {row['load_s']} "
              f"| {row['first_run_s']} | {row['mean_ms']} | {row['min_ms']} | {row['cos_vs_cpu_fp32']} |")


if __name__ == "__main__":
    main()
