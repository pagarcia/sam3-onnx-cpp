"""Share byte-identical ONNX tensors in a separate, lossless storage candidate.

Includes Constant tensor attributes, not just initializers. Nodes, precision,
interfaces and tensor values remain unchanged. Source files are never edited.
Runtime loading and timing still need qualification before adopting a candidate.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path

import onnx
from onnx.external_data_helper import set_external_data


def tensors(graph):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR: yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS: yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH: yield from tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs: yield from tensors(child)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def semantic_digest(model):
    # These two fields only locate storage. Loading must have restored all raw
    # bytes before they are cleared; everything else is compared exactly.
    for tensor in tensors(model.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError('Cannot compare unresolved external tensors')
        tensor.ClearField('data_location')
        tensor.ClearField('external_data')
    return hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()


def compact(source, destination, *, weight_name=None, alignment=65536, threshold=4096):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source.parent or destination.is_relative_to(source.parent):
        raise ValueError('Keep candidates outside the source model directory')
    if alignment <= 0 or alignment & (alignment-1) or threshold < 1:
        raise ValueError('Positive power-of-two alignment and threshold required')
    name = weight_name or (source.name + '.data')
    if Path(name).name != name or name in ('.', '..', source.name) or ':' in name or '\\' in name:
        raise ValueError('Weight companion must be a distinct basename')
    # Validate external paths before allowing the ONNX loader to open them.
    header = onnx.load_model(source, load_external_data=False)
    source_files = {source}
    for tensor in tensors(header.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            fields = {e.key:e.value for e in tensor.external_data}
            path = (source.parent / fields['location']).resolve()
            if not path.is_relative_to(source.parent): raise ValueError('External data leaves source directory')
            source_files.add(path)
    del header
    original = [{'path':p.name, 'size':p.stat().st_size, 'sha256':digest(p)} for p in sorted(source_files)]
    model = onnx.load_model(source)
    before = semantic_digest(model)
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / source.name
    unique = {}
    references = padding = moved = 0
    with (destination / name).open('xb') as weights:
        for tensor in tensors(model.graph):
            raw = tensor.raw_data
            if len(raw) < threshold: continue
            key = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(raw).digest())
            if key not in unique:
                gap = (-weights.tell()) % alignment
                weights.write(b'\0'*gap); padding += gap
                unique[key] = (weights.tell(), len(raw))
                weights.write(raw)
            offset, length = unique[key]
            set_external_data(tensor, location=name, offset=offset, length=length)
            tensor.ClearField('raw_data')
            references += 1; moved += length
    target.write_bytes(model.SerializeToString(deterministic=True))
    del model
    gc.collect()
    restored = onnx.load_model(target)
    after = semantic_digest(restored)
    del restored
    if before != after: raise ValueError('Reloaded graph or tensor bytes changed')
    for entry in original:
        if digest(source.parent / entry['path']) != entry['sha256']: raise ValueError('Source changed during compaction')
    outputs = [{'path':p.name, 'size':p.stat().st_size, 'sha256':digest(p)} for p in (target, destination/name)]
    result = dict(schema=1, method='exact tensor storage sharing', alignment=alignment, threshold=threshold,
        source=original, outputs=outputs, unique_tensors=len(unique), references=references,
        padding_bytes=padding, repeated_tensor_bytes=moved-sum(length for _,length in unique.values()),
        saved_bytes=sum(e['size'] for e in original)-sum(e['size'] for e in outputs),
        reloaded_semantic_sha256=after, exact_graph_and_tensor_identity=True, runtime_qualified=False)
    (destination/'storage.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight-name')
    args = parser.parse_args()
    print(json.dumps(compact(args.source, args.output, weight_name=args.weight_name), indent=2))
