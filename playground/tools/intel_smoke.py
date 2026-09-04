import openvino as ov


left = ov.opset13.parameter([1, 16], ov.Type.f32, name="left")
right = ov.opset13.parameter([1, 16], ov.Type.f32, name="right")
model = ov.Model([ov.opset13.add(left, right)], [left, right], "npu_agent_smoke")
compiled = ov.Core().compile_model(
    model,
    "NPU",
    {"NPU_COMPILER_TYPE": "PLUGIN", "NPU_PLATFORM": "4000"},
)
blob = compiled.export_model()
if hasattr(blob, "getvalue"):
    blob = blob.getvalue()
if not isinstance(blob, bytes) or not blob:
    raise RuntimeError("Intel NPU compiler returned an empty blob")
