"""Build an isolated, experimental weight-only INT8 GPU encoder candidate.

FP16 activations and original operators remain. Selected constant MatMul weights
are stored as INT8, dequantized with standard ONNX operators and cast to FP16.
This changes numerical weights and is NOT a release export. Measure segmentation
agreement, provider placement, loading time and memory before considering use.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transform(model):
    """Per-output-column symmetric INT8; no activation quantization/calibration."""
    initializers = {t.name: t for t in model.graph.initializer}
    consumers = collections.defaultdict(list)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            consumers[name].append((node.op_type, index))
    selected = [t for t in model.graph.initializer if t.data_type == TensorProto.FLOAT16
                and len(t.dims) == 2 and consumers[t.name]
                and all(op == "MatMul" and index == 1 for op, index in consumers[t.name])]
    used_names = set(initializers) | {n for node in model.graph.node for n in (*node.input, *node.output)}
    additions, nodes = [], []
    error_squared = reference_squared = 0.0
    largest_error = 0.0
    original_bytes = quantized_bytes = 0
    for tensor in selected:
        name = tensor.name
        qname, sname, zname, dname = [name + suffix for suffix in
                                    ("__storage_i8", "__storage_scale", "__storage_zero", "__storage_f32")]
        if any(n in used_names for n in (qname, sname, zname, dname)):
            raise ValueError(f"Generated name collision: {name}")
        used_names.update((qname, sname, zname, dname))
        weights = numpy_helper.to_array(tensor).astype(np.float32)
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite source weights: {name}")
        maximum = np.abs(weights).max(axis=0)
        scale = np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float32)
        quantized = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
        zero = np.zeros(scale.shape, dtype=np.int8)
        restored = (quantized.astype(np.float32) * scale).astype(np.float16).astype(np.float32)
        error = restored - weights
        error_squared += float(np.square(error, dtype=np.float64).sum())
        reference_squared += float(np.square(weights, dtype=np.float64).sum())
        largest_error = max(largest_error, float(np.abs(error).max()))
        original_bytes += weights.size * 2
        quantized_bytes += quantized.nbytes + scale.nbytes + zero.nbytes
        additions.extend((numpy_helper.from_array(quantized, qname), numpy_helper.from_array(scale, sname),
                          numpy_helper.from_array(zero, zname)))
        nodes.append(helper.make_node("DequantizeLinear", [qname, sname, zname], [dname], axis=1,
                                      name=qname + "_decode"))
        nodes.append(helper.make_node("Cast", [dname], [name], to=TensorProto.FLOAT16,
                                      name=qname + "_fp16"))
    selected_names = {t.name for t in selected}
    keep = [t for t in model.graph.initializer if t.name not in selected_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep + additions)
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes + original_nodes)
    return {"matrices": len(selected), "original_selected_bytes": original_bytes,
            "stored_selected_bytes": quantized_bytes, "maximum_weight_error": largest_error,
            "relative_weight_l2": float(np.sqrt(error_squared / reference_squared)) if reference_squared else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory, never a deployed model directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = onnx.load_model(args.source)
    metrics = transform(model)
    if metrics["matrices"] == 0:
        raise ValueError("No eligible FP16 MatMul weights")
    target = args.output / args.source.name
    onnx.save_model(model, target, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=target.name + "_data", size_threshold=1024)
    onnx.checker.check_model(str(target))
    record = {"schema": 1, "release_qualified": False, "method": "weight-only INT8 + FP16 activations",
              "onnx": onnx.__version__, "source_sha256": digest(args.source), "weight_metrics": metrics,
              "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                        for p in sorted(args.output.iterdir()) if p.is_file()]}
    (args.output / "experiment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
