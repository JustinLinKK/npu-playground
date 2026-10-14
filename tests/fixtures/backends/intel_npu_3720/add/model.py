import numpy as np
import openvino as ov
from openvino import opset13 as ops


def build_model(manifest):
    inputs = []
    for tensor in manifest['tensors']:
        if tensor['direction'] == 'input':
            node = ops.parameter(tensor['shape'], np.dtype(tensor['dtype']), name=tensor['name'])
            node.output(0).get_tensor().set_names({tensor['name']})
            inputs.append(node)
    result = ops.add(*inputs)
    result.output(0).get_tensor().set_names({'output'})
    return ov.Model([result], inputs)
