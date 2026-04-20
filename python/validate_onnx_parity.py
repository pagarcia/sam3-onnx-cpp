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
    _load_prompt_spec,
    _parse_box_text,
    _parse_points_text,
    run_compare,
)


def _load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "cases" in payload:
        cases = payload["cases"]
    elif isinstance(payload, list):
        cases = payload
    else:
        raise SystemExit("Validation manifest must be a list or an object with a 'cases' field.")

    if not cases:
        raise SystemExit("Validation manifest did not contain any cases.")
    return cases


def _resolve_prompt_spec(case: dict):
    if "prompt" in case and isinstance(case["prompt"], dict):
        return case["prompt"]
    if "prompt_json" in case and case["prompt_json"]:
        return _load_prompt_spec(Path(case["prompt_json"]).resolve())
    if "box" in case and case["box"]:
        return {"prompt": "bounding_box", "box": list(_parse_box_text(case["box"]))}
    if "points" in case and case["points"]:
        points, labels = _parse_points_text(case["points"])
        return {
            "prompt": "seed_points",
            "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
        }
    raise SystemExit(
        "Each validation case must provide one of: prompt object, prompt_json, points, or box."
    )


def _check_thresholds(summary: dict, thresholds: dict) -> tuple[bool, list[dict]]:
    failures = []
    for metric_name, min_value in thresholds.get("min", {}).items():
        actual = float(summary.get(metric_name, 0.0))
        if actual < float(min_value):
            failures.append(
                {
                    "metric": metric_name,
                    "kind": "min",
                    "expected": float(min_value),
                    "actual": actual,
                }
            )
    for metric_name, max_value in thresholds.get("max", {}).items():
        actual = float(summary.get(metric_name, 0.0))
        if actual > float(max_value):
            failures.append(
                {
                    "metric": metric_name,
                    "kind": "max",
                    "expected": float(max_value),
                    "actual": actual,
                }
            )
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed native-vs-ONNX validation cases and gate them with explicit thresholds."
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="JSON manifest containing a list of parity cases or an object with a top-level 'cases' field.",
    )
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
        "--max_frames",
        type=int,
        default=20,
        help="Default frame limit for cases that do not override max_frames.",
    )
    parser.add_argument(
        "--onnx_max_mem_frames",
        type=int,
        default=2,
        help="Default ONNX spatial-memory cap for cases that do not override it.",
    )
    parser.add_argument(
        "--onnx_max_obj_ptrs",
        type=int,
        default=16,
        help="Default ONNX object-pointer cap for cases that do not override it.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory for the validation summaries and per-case dumps.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations in the ONNX subprocess.",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    cases = _load_cases(cases_path)
    outdir = Path(args.outdir).resolve() if args.outdir else Path(
        tempfile.mkdtemp(prefix="sam3_validate_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
    )
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    all_passed = True

    print(f"[INFO] Validation manifest: {cases_path}")
    print(f"[INFO] Output dir: {outdir}")

    for idx, case in enumerate(cases, start=1):
        case_name = case.get("name", f"case_{idx:02d}")
        case_dir = outdir / case_name
        prompt_spec = _resolve_prompt_spec(case)
        print(f"[INFO] Case {idx}/{len(cases)}: {case_name}")

        run = run_compare(
            video_path=str(Path(case["video"]).resolve()),
            onnx_dir=Path(case.get("onnx_dir", args.onnx_dir)).resolve(),
            checkpoint=str(Path(case.get("checkpoint", args.checkpoint)).resolve()),
            sam3_repo=Path(case.get("sam3_repo", args.sam3_repo)).resolve(),
            prompt_spec=prompt_spec,
            outdir=case_dir,
            max_frames=int(case.get("max_frames", args.max_frames)),
            safe=bool(case.get("safe", args.safe)),
            onnx_accel=case.get("onnx_accel", args.onnx_accel),
            onnx_max_mem_frames=int(case.get("onnx_max_mem_frames", args.onnx_max_mem_frames)),
            onnx_max_obj_ptrs=int(case.get("onnx_max_obj_ptrs", args.onnx_max_obj_ptrs)),
            onnx_variant=case.get("onnx_variant", args.onnx_variant),
        )
        summary = run["summary"]
        passed, failures = _check_thresholds(summary, case.get("thresholds", {}))
        all_passed = all_passed and passed
        results.append(
            {
                "name": case_name,
                "video": str(Path(case["video"]).resolve()),
                "prompt": prompt_spec,
                "passed": passed,
                "failures": failures,
                "summary_path": str(run["summary_json"]),
                "summary": summary,
            }
        )

        status = "PASS" if passed else "FAIL"
        print(
            f"[INFO] {status} | mean_iou={summary['mean_iou']:.4f} | "
            f"min_iou={summary['min_iou']:.4f} | onnx_mean_total={summary['onnx_mean_total_ms']:.1f} ms/frame"
        )
        if failures:
            for failure in failures:
                print(
                    f"[WARN] {case_name}: {failure['metric']} violated {failure['kind']} threshold "
                    f"(expected {failure['kind']}={failure['expected']}, actual={failure['actual']:.4f})"
                )

    payload = {
        "manifest": str(cases_path),
        "passed": all_passed,
        "case_count": len(results),
        "results": results,
    }
    summary_path = outdir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Validation summary: {summary_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
