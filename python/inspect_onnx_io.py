# sam3-onnx-cpp/python/inspect_onnx_io.py
#!/usr/bin/env python3
import os
from pathlib import Path
import onnxruntime as ort


def dump(path: Path):
    print("\n===", path.name, "===")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("Providers:", sess.get_providers())
    print("Inputs:")
    for i in sess.get_inputs():
        print(" -", i.name, i.type, i.shape)
    print("Outputs:")
    for o in sess.get_outputs():
        print(" -", o.name, o.type, o.shape)


def main():
    root = Path(__file__).resolve().parent.parent
    onnx_dir = root / "checkpoints" / "sam3" / "onnx"

    variant = os.getenv("SAM3_ONNX_VARIANT", "").strip().lower()
    if variant not in ("fp16", "fp32"):
        variant = "fp32"

    # Pick requested variant, but fall back if not present
    if variant == "fp16":
        enc = onnx_dir / "vision_encoder_fp16.onnx"
        dec = onnx_dir / "prompt_encoder_mask_decoder_fp16.onnx"
        if not enc.exists():
            enc = onnx_dir / "vision_encoder.onnx"
        if not dec.exists():
            dec = onnx_dir / "prompt_encoder_mask_decoder.onnx"
    else:
        enc = onnx_dir / "vision_encoder.onnx"
        dec = onnx_dir / "prompt_encoder_mask_decoder.onnx"
        if not enc.exists():
            enc = onnx_dir / "vision_encoder_fp16.onnx"
        if not dec.exists():
            dec = onnx_dir / "prompt_encoder_mask_decoder_fp16.onnx"

    if not enc.exists() or not dec.exists():
        raise FileNotFoundError(
            f"Could not find encoder/decoder ONNX in: {onnx_dir}\n"
            f"Expected (fp32): vision_encoder.onnx + prompt_encoder_mask_decoder.onnx\n"
            f"or (fp16): vision_encoder_fp16.onnx + prompt_encoder_mask_decoder_fp16.onnx"
        )

    print(f"[INFO] SAM3_ONNX_VARIANT={os.getenv('SAM3_ONNX_VARIANT','')!r} (effective: {variant})")
    dump(enc)
    dump(dec)


if __name__ == "__main__":
    main()