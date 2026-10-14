import numpy as np
import openvino as ov
from openvino import opset13 as ops


def build_model(manifest):
    x = ops.parameter(manifest['tensors'][0]['shape'], np.float32, name='x')
    x.output(0).get_tensor().set_names({'x'})
    result = ops.reduce_sum(x, ops.constant([0], np.int64), keep_dims=True)
    result.output(0).get_tensor().set_names({'output'})
    return ov.Model([result], [x])
