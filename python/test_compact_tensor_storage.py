import tempfile
from pathlib import Path
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from compact_tensor_storage import compact


class CompactStorageTests(unittest.TestCase):
    def test_initializer_and_constant_aliases_preserve_inference_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            values = np.arange(2048, dtype=np.float32)
            graph = helper.make_graph([
                helper.make_node('Constant', [], ['constant'], value=numpy_helper.from_array(values)),
                helper.make_node('Add', ['input','weights'], ['first']),
                helper.make_node('Add', ['first','constant'], ['output'])], 'storage',
                [helper.make_tensor_value_info('input',TensorProto.FLOAT,[2048])],
                [helper.make_tensor_value_info('output',TensorProto.FLOAT,[2048])],
                [numpy_helper.from_array(values,'weights')])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',18)], ir_version=9)
            path = source/'test.onnx'; onnx.save(model,path)
            candidate=root/'candidate'; result=compact(path,candidate,alignment=4096)
            self.assertEqual(result['unique_tensors'],1)
            self.assertEqual(result['references'],2)
            self.assertTrue(result['exact_graph_and_tensor_identity'])
            feed={'input': np.linspace(-1,1,2048,dtype=np.float32)}
            before=ort.InferenceSession(str(path), providers=['CPUExecutionProvider']).run(None,feed)[0]
            after=ort.InferenceSession(str(candidate/path.name), providers=['CPUExecutionProvider']).run(None,feed)[0]
            np.testing.assert_array_equal(before,after)
            # External-data input also roundtrips and never modifies its sources.
            second=compact(candidate/path.name,root/'second',alignment=4096)
            self.assertEqual(result['reloaded_semantic_sha256'],second['reloaded_semantic_sha256'])
            with self.assertRaises(ValueError): compact(path,source/'unsafe')
            with self.assertRaises(ValueError): compact(path,root/'bad',weight_name='../outside')
            with self.assertRaises(FileExistsError): compact(path,candidate)


if __name__=='__main__': unittest.main()
