# sam3-onnx-cpp/python/inspect_onnx_io.py
#!/usr/bin/env python3
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
    dump(onnx_dir / "vision_encoder.onnx")
    dump(onnx_dir / "prompt_encoder_mask_decoder.onnx")

if __name__ == "__main__":
    main()
