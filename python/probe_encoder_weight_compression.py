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
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transform(model, *, include_node_regex=None, exclude_node_regex=None):
    """Select by consuming node name; weight error is not segmentation accuracy.

    All consumers must match include and none may match exclude. An initializer
    shared with another operator or exposed as an input/output is never changed.
    """
    include = re.compile(include_node_regex) if include_node_regex is not None else None
    exclude = re.compile(exclude_node_regex) if exclude_node_regex is not None else None
    if any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in model.graph.node for attribute in node.attribute):
        raise ValueError("Subgraph consumers are not supported by this experiment")
    initializers = {t.name: t for t in model.graph.initializer}
    consumers = collections.defaultdict(list)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            consumers[name].append((node, index))
    interface_names = {value.name for value in (*model.graph.input, *model.graph.output)}
    selected = [t for t in model.graph.initializer if t.data_type == TensorProto.FLOAT16
                and len(t.dims) == 2 and all(t.dims) and consumers[t.name]
                and t.name not in interface_names
                and all(node.op_type == "MatMul" and node.domain in ("", "ai.onnx") and index == 1
                        and (include is None or include.search(node.name))
                        and (exclude is None or not exclude.search(node.name))
                        for node, index in consumers[t.name])]
    used_names = set(initializers) | {n for node in model.graph.node for n in (*node.input, *node.output)}
    additions, nodes = [], []
    error_squared = reference_squared = 0.0
    largest_error = 0.0
    original_bytes = quantized_bytes = 0
    matrix_metrics = []
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
        tensor_error = float(np.square(error, dtype=np.float64).sum())
        tensor_reference = float(np.square(weights, dtype=np.float64).sum())
        maximum_error = float(np.abs(error).max())
        error_squared += tensor_error
        reference_squared += tensor_reference
        largest_error = max(largest_error, maximum_error)
        original_bytes += weights.size * 2
        quantized_bytes += quantized.nbytes + scale.nbytes + zero.nbytes
        matrix_metrics.append({"name": name, "shape": list(tensor.dims),
                               "consuming_nodes": [node.name for node, _ in consumers[name]],
                               "original_bytes": weights.size * 2,
                               "stored_bytes": quantized.nbytes + scale.nbytes + zero.nbytes,
                               "maximum_weight_error": maximum_error,
                               "error_squared": tensor_error, "reference_squared": tensor_reference,
                               "relative_weight_l2": float(np.sqrt(tensor_error / tensor_reference))
                               if tensor_reference else 0.0})
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
            "relative_weight_l2": float(np.sqrt(error_squared / reference_squared)) if reference_squared else 0.0,
            "per_matrix": matrix_metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory, never a deployed model directory")
    parser.add_argument("--include-node-regex", help="Quantize weights only if ALL consuming node names match")
    parser.add_argument("--exclude-node-regex", help="Keep original weights if ANY consuming node name matches")
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output == args.source.parent or args.output.is_relative_to(args.source.parent):
        raise ValueError("Keep candidates outside the source model directory")
    for pattern in (args.include_node_regex, args.exclude_node_regex):
        if pattern is not None:
            re.compile(pattern)
    args.output.mkdir(parents=True, exist_ok=False)
    model = onnx.load_model(args.source)
    metrics = transform(model, include_node_regex=args.include_node_regex, exclude_node_regex=args.exclude_node_regex)
    if metrics["matrices"] == 0:
        raise ValueError("No eligible FP16 MatMul weights")
    target = args.output / args.source.name
    onnx.save_model(model, target, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=target.name + "_data", size_threshold=1024)
    onnx.checker.check_model(str(target))
    record = {"schema": 2, "release_qualified": False, "method": "weight-only INT8 + FP16 activations",
              "selection": {"include_node_regex": args.include_node_regex,
                            "exclude_node_regex": args.exclude_node_regex},
              "quality_and_runtime_qualification": "pending; weight error is not segmentation accuracy",
              "onnx": onnx.__version__, "source_sha256": digest(args.source), "weight_metrics": metrics,
              "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                        for p in sorted(args.output.iterdir()) if p.is_file()]}
    (args.output / "experiment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
