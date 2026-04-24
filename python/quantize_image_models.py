#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_onnx_dir() -> Path:
    return repo_root() / "checkpoints" / "sam3" / "onnx"


def model_paths(onnx_dir: Path, model: str) -> tuple[Path, Path]:
    if model == "encoder":
        return onnx_dir / "vision_encoder.onnx", onnx_dir / "vision_encoder.int8.onnx"
    if model == "decoder":
        return onnx_dir / "prompt_encoder_mask_decoder.onnx", onnx_dir / "prompt_encoder_mask_decoder.int8.onnx"
    raise ValueError(f"Unsupported model kind: {model}")


def default_op_types(model: str) -> list[str]:
    if model == "encoder":
        # SAM3's vision encoder is transformer-heavy. On CPU, quantizing
        # Gather alongside MatMul gave a better speed/compatibility tradeoff
        # than MatMul alone, while ConvInteger was not runnable here.
        return ["MatMul", "Gather"]
    if model == "decoder":
        return ["MatMul", "Gemm"]
    raise ValueError(f"Unsupported model kind: {model}")


def external_data_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + "_data")


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def remove_artifact_pair(model_path: Path) -> None:
    for candidate in (model_path, external_data_path(model_path)):
        if candidate.exists():
            remove_path(candidate)


def ensure_input_ready(model_path: Path) -> None:
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}")

    sidecar = external_data_path(model_path)
    if sidecar.exists():
        return

    print(
        f"[WARN] {sidecar.name} is missing next to {model_path.name}. "
        "This is only okay if the model is fully embedded."
    )


def preprocess_model(input_path: Path, workdir: Path) -> Path:
    output_path = workdir / input_path.name
    previous_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        quant_pre_process(
            input_model=input_path,
            output_model_path=output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            external_data_location=output_path.name + "_data",
            external_data_size_threshold=0,
        )
    finally:
        os.chdir(previous_cwd)
    return output_path


def quantize_model(input_path: Path,
                   output_path: Path,
                   model_kind: str,
                   do_preprocess: bool,
                   op_types: list[str],
                   allow_preprocess_fallback: bool = True) -> dict[str, object]:
    ensure_input_ready(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_artifact_pair(output_path)

    with tempfile.TemporaryDirectory(prefix=f"sam3_quant_{input_path.stem}_") as tmp:
        workdir = Path(tmp)
        preprocess_used = False
        model_for_quant = input_path
        if do_preprocess:
            try:
                model_for_quant = preprocess_model(input_path, workdir)
                preprocess_used = True
            except Exception as exc:
                if not allow_preprocess_fallback:
                    raise
                print(
                    f"[WARN] Preprocess failed for {input_path.name}; "
                    f"falling back to direct quantization: {exc}"
                )

        quantize_dynamic(
            model_input=model_for_quant,
            model_output=output_path,
            op_types_to_quantize=op_types,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            use_external_data_format=True,
            extra_options={
                "MatMulConstBOnly": True,
            },
        )
    return {
        "preprocess_requested": do_preprocess,
        "preprocess_used": preprocess_used,
        "op_types": list(op_types),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CPU-friendly int8 SAM3 image ONNX artifacts."
    )
    parser.add_argument(
        "--onnx_dir",
        default=str(default_onnx_dir()),
        help="Directory containing the downloaded SAM3 image ONNX files.",
    )
    parser.add_argument(
        "--model",
        choices=["encoder", "decoder", "both"],
        default="encoder",
        help="Which image model(s) to quantize.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run ONNX Runtime preprocessing before quantization.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing int8 output files.",
    )
    parser.add_argument(
        "--ops",
        default="",
        help="Optional comma-separated op types override, for example MatMul,Gather.",
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir).resolve()
    targets = ["encoder", "decoder"] if args.model == "both" else [args.model]

    for target in targets:
        input_path, output_path = model_paths(onnx_dir, target)
        if output_path.exists() and not args.force:
            raise SystemExit(
                f"{output_path} already exists. Re-run with --force to overwrite it."
            )

        op_types = (
            [part.strip() for part in args.ops.split(",") if part.strip()]
            if args.ops.strip()
            else default_op_types(target)
        )

        print(f"[INFO] Quantizing {target}:")
        print(f"       input  = {input_path}")
        print(f"       output = {output_path}")
        print(f"       ops    = {','.join(op_types)}")
        result = quantize_model(
            input_path,
            output_path,
            target,
            do_preprocess=args.preprocess,
            op_types=op_types,
        )

        sidecar = external_data_path(output_path)
        print(f"[OK] Wrote {output_path}")
        if sidecar.exists():
            print(f"[OK] Wrote {sidecar}")
        if result["preprocess_requested"]:
            print(
                "[INFO] preprocess_used="
                + ("yes" if result["preprocess_used"] else "no (fell back to direct quantization)")
            )


if __name__ == "__main__":
    main()
