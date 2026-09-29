import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

from probe_encoder_weight_compression import transform


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
        reference = ort.InferenceSession(original, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
        candidate = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
        self.assertEqual(candidate.dtype, np.float16)
        np.testing.assert_allclose(candidate, reference, rtol=.02, atol=.003)
        self.assertEqual(float(candidate[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
