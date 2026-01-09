# sam3-onnx-cpp/python/inspect_onnx_io.py
#!/usr/bin/env python3
import os
from pathlib import Path
import onnxruntime as ort


def dump(path: Path):
    print("\n===", path.name, "===")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("Providers (forced):", sess.get_providers())
    print("Inputs:")
    for i in sess.get_inputs():
        print(" -", i.name, i.type, i.shape)
    print("Outputs:")
    for o in sess.get_outputs():
        print(" -", o.name, o.type, o.shape)


def main():
    root = Path(__file__).resolve().parent.parent
    onnx_dir = root / "checkpoints" / "sam3" / "onnx"

    av = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in av

    requested = os.getenv("SAM3_ONNX_VARIANT", "").strip().lower()
    if requested not in ("fp16", "fp32"):
        accel = os.getenv("SAM3_ORT_ACCEL", "auto").strip().lower()
        requested = "fp16" if (accel == "cuda" or (accel == "auto" and cuda_available)) else "fp32"

    def pick(primary: Path, fallback: Path) -> Path:
        return primary if primary.exists() else fallback

    if requested == "fp16":
        enc = pick(onnx_dir / "vision_encoder_fp16.onnx", onnx_dir / "vision_encoder.onnx")
        dec = pick(onnx_dir / "prompt_encoder_mask_decoder_fp16.onnx", onnx_dir / "prompt_encoder_mask_decoder.onnx")
    else:
        enc = pick(onnx_dir / "vision_encoder.onnx", onnx_dir / "vision_encoder_fp16.onnx")
        dec = pick(onnx_dir / "prompt_encoder_mask_decoder.onnx", onnx_dir / "prompt_encoder_mask_decoder_fp16.onnx")

    if not enc.exists() or not dec.exists():
        raise FileNotFoundError(
            f"Could not find encoder/decoder ONNX in: {onnx_dir}\n"
            f"Tip: run .\\fetch_onnx_models.bat fp32 or fp16"
        )

    effective = "fp16" if "fp16" in enc.name.lower() else "fp32"
    print(f"[INFO] ORT providers available: {av}")
    print(f"[INFO] Requested variant: {requested} | Effective: {effective}")
    print(f"[INFO] Encoder: {enc.name}")
    print(f"[INFO] Decoder: {dec.name}")

    dump(enc)
    dump(dec)


if __name__ == "__main__":
    main()
