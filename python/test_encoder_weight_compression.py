import unittest
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

from probe_encoder_weight_compression import transform


def cpu_session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])


class EncoderStorageTests(unittest.TestCase):
    def test_candidate_keeps_fp16_interface_and_handles_zero_columns(self):
        weights = np.arange(128, dtype=np.float16).reshape(8, 16) / 128
        weights[:, 0] = 0
        graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16])],
            [numpy_helper.from_array(weights, "w")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        original = model.SerializeToString()
        metrics = transform(model)
        self.assertEqual(metrics["matrices"], 1)
        onnx.checker.check_model(model)
        feeds = {"x": np.linspace(-1, 1, 8, dtype=np.float16)[None]}
        reference = cpu_session(original).run(None, feeds)[0]
        candidate = cpu_session(model.SerializeToString()).run(None, feeds)[0]
        self.assertEqual(candidate.dtype, np.float16)
        np.testing.assert_allclose(candidate, reference, rtol=.02, atol=.003)
        self.assertEqual(float(candidate[0, 0]), 0.0)

    def selection_model(self):
        rng = np.random.default_rng(19)
        nodes, weights = [], []
        current = "x"
        for index, kind in enumerate(("attention", "mlp", "mlp")):
            name, output = f"w{index}", f"a{index}"
            weights.append(numpy_helper.from_array(rng.normal(0, .1, (8, 8)).astype(np.float16), name))
            nodes.append(helper.make_node("MatMul", [current, name], [output],
                                          name=f"layers.{index}/{kind}/MatMul"))
            current = output
        return helper.make_model(helper.make_graph(nodes, "selection",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])],
            [helper.make_tensor_value_info(current, TensorProto.FLOAT16, [1, 8])], weights),
            opset_imports=[helper.make_opsetid("", 18)], ir_version=9)

    def test_selective_candidate_retains_excluded_weights_and_ops(self):
        model = self.selection_model()
        before = {t.name: t.SerializeToString() for t in model.graph.initializer}
        original_nodes = [n.SerializeToString() for n in model.graph.node]
        reference = onnx.ModelProto.FromString(model.SerializeToString())
        metrics = transform(model, include_node_regex="/mlp/", exclude_node_regex=r"layers\.2/")
        self.assertEqual(metrics["matrices"], 1)
        self.assertEqual(metrics["per_matrix"][0]["name"], "w1")
        after = {t.name: t.SerializeToString() for t in model.graph.initializer}
        for name in ("w0", "w2"):
            self.assertEqual(after[name], before[name])
        self.assertEqual([n.SerializeToString() for n in model.graph.node[-3:]], original_nodes)
        self.assertNotIn("w1", after)
        # Inspect the reconstructed weights exactly: an end-to-end tolerance
        # alone could hide a wrong axis. CPU FP16 promotion can vary rounding in
        # subsequent MatMul results, even with graph optimizations disabled.
        tensor = reference.graph.initializer[1]
        w = numpy_helper.to_array(tensor).astype(np.float32)
        scale = np.abs(w).max(axis=0) / 127
        restored = (np.clip(np.rint(w / scale), -127, 127).astype(np.int8).astype(np.float32) * scale).astype(np.float16)
        tensor.CopyFrom(numpy_helper.from_array(restored, tensor.name))
        model.graph.output.append(helper.make_tensor_value_info("w1", TensorProto.FLOAT16, [8, 8]))
        onnx.checker.check_model(model)
        feeds = {"x": np.linspace(-1, 1, 8, dtype=np.float16)[None]}
        expected = cpu_session(reference.SerializeToString()).run(None, feeds)[0]
        actual, reconstructed = cpu_session(model.SerializeToString()).run(None, feeds)
        np.testing.assert_array_equal(reconstructed, restored)
        np.testing.assert_allclose(actual, expected, rtol=.001, atol=.00001)

    def test_all_consumers_must_be_selected(self):
        model = self.selection_model()
        model.graph.node[2].input[1] = "w1"
        before = model.SerializeToString()
        metrics = transform(model, include_node_regex="/mlp/", exclude_node_regex=r"layers\.2/")
        self.assertEqual(metrics["matrices"], 0)
        self.assertEqual(model.SerializeToString(), before)

    def test_interface_initializer_is_not_rewritten(self):
        model = self.selection_model()
        model.graph.input.append(helper.make_tensor_value_info("w1", TensorProto.FLOAT16, [8, 8]))
        metrics = transform(model, include_node_regex=r"layers\.1/")
        self.assertEqual(metrics["matrices"], 0)

    def test_invalid_pattern_does_not_mutate_model(self):
        model = self.selection_model()
        before = model.SerializeToString()
        with self.assertRaises(re.error):
            transform(model, include_node_regex="[")
        self.assertEqual(model.SerializeToString(), before)

    def test_subgraph_consumer_is_rejected_before_mutation(self):
        model = self.selection_model()
        body = helper.make_graph([helper.make_node("Identity", ["w1"], ["branch_w"])],
                                 "branch", [], [helper.make_tensor_value_info("branch_w", TensorProto.FLOAT16, [8, 8])])
        model.graph.node.append(helper.make_node("If", ["condition"], ["result"], then_branch=body, else_branch=body))
        before = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, "Subgraph"):
            transform(model, include_node_regex="/mlp/")
        self.assertEqual(model.SerializeToString(), before)


if __name__ == "__main__":
    unittest.main()
