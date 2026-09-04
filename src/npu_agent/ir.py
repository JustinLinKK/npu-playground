from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import ArgumentDirection, KernelIR, KernelManifest


@dataclass(frozen=True)
class IRValidationResult:
    passed: bool
    mismatches: dict[str, float]
    message: str


def generate_inputs(manifest: KernelManifest) -> dict[str, np.ndarray | int | float | bool]:
    rng = np.random.default_rng(manifest.seed)
    values: dict[str, np.ndarray | int | float | bool] = {}
    for tensor in manifest.tensors:
        if tensor.direction == ArgumentDirection.OUTPUT:
            continue
        dtype = np.dtype(tensor.dtype)
        if np.issubdtype(dtype, np.integer):
            value = rng.integers(-3, 4, size=tensor.shape, dtype=dtype)
        else:
            value = rng.normal(0.0, 0.5, size=tensor.shape).astype(dtype)
        values[tensor.name] = value
    for scalar in manifest.scalars:
        values[scalar.name] = scalar.value
    return values


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value.astype(np.float64) - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(value.dtype)


def _conv2d(value: np.ndarray, weight: np.ndarray, padding: int) -> np.ndarray:
    n, channels, height, width = value.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    if channels != weight_channels:
        raise ValueError("conv2d channel mismatch")
    padded = np.pad(value, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.zeros((n, out_channels, height, width), dtype=np.float32)
    for batch in range(n):
        for out_channel in range(out_channels):
            for row in range(height):
                for col in range(width):
                    region = padded[batch, :, row : row + kernel_h, col : col + kernel_w]
                    output[batch, out_channel, row, col] = np.sum(
                        region.astype(np.float32) * weight[out_channel].astype(np.float32)
                    )
    return output.astype(value.dtype)


def _moving_average(value: np.ndarray, window: int) -> np.ndarray:
    output = np.empty_like(value)
    for index in range(value.shape[0]):
        start = max(0, index - window + 1)
        output[index] = np.mean(value[start : index + 1], dtype=np.float64)
    return output


def evaluate_operation(operation: str, inputs: list[Any], attributes: dict[str, Any]) -> np.ndarray:
    if operation == "add":
        return np.asarray(inputs[0]) + np.asarray(inputs[1])
    if operation == "reduce_sum":
        axis = attributes.get("axis")
        return np.asarray(inputs[0]).sum(
            axis=axis,
            dtype=np.float32,
            keepdims=bool(attributes.get("keepdims", False)),
        )
    if operation == "matmul":
        return np.matmul(np.asarray(inputs[0]).astype(np.float32), np.asarray(inputs[1]).astype(np.float32)).astype(
            attributes.get("output_dtype", "float16")
        )
    if operation == "transpose":
        return np.transpose(np.asarray(inputs[0]), axes=attributes.get("axes"))
    if operation == "softmax":
        return _softmax(np.asarray(inputs[0]), int(attributes.get("axis", -1)))
    if operation == "layer_norm":
        value, weight, bias = (np.asarray(item) for item in inputs[:3])
        axis = int(attributes.get("axis", -1))
        epsilon = float(attributes.get("epsilon", 1e-5))
        mean = value.astype(np.float64).mean(axis=axis, keepdims=True)
        variance = ((value.astype(np.float64) - mean) ** 2).mean(axis=axis, keepdims=True)
        return (((value - mean) / np.sqrt(variance + epsilon)) * weight + bias).astype(value.dtype)
    if operation == "attention":
        query, key, value = (np.asarray(item) for item in inputs[:3])
        scale = float(attributes.get("scale", query.shape[-1] ** -0.5))
        scores = np.matmul(query.astype(np.float32), np.swapaxes(key.astype(np.float32), -1, -2)) * scale
        probabilities = _softmax(scores, -1)
        return np.matmul(probabilities.astype(np.float32), value.astype(np.float32)).astype(query.dtype)
    if operation == "moving_average":
        return _moving_average(np.asarray(inputs[0]), int(attributes["window"]))
    if operation == "conv2d":
        return _conv2d(np.asarray(inputs[0]), np.asarray(inputs[1]), int(attributes.get("padding", 0)))
    if operation == "gamma_correction":
        value = np.asarray(inputs[0])
        gamma = float(attributes["gamma"])
        return np.power(np.clip(value, 0, 1).astype(np.float64), gamma).astype(value.dtype)
    raise ValueError(f"unsupported IR operation: {operation}")


def execute_ir(kernel_ir: KernelIR, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    values = dict(inputs)
    for operation in kernel_ir.operations:
        args = [values[name] for name in operation.inputs]
        attributes = operation.attributes.model_dump(exclude_none=True)
        values[operation.output] = evaluate_operation(operation.op, args, attributes)
    return {output.name: np.asarray(values[output.name], dtype=output.dtype) for output in kernel_ir.outputs}


def execute_oracle(manifest: KernelManifest, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    operation = manifest.oracle.operation
    op_name = {"vector_add": "add"}.get(operation, operation)
    input_names = [arg.name for arg in manifest.tensors if arg.direction != ArgumentDirection.OUTPUT]
    scalar_names = [arg.name for arg in manifest.scalars]
    arguments = [inputs[name] for name in input_names + scalar_names]
    output = evaluate_operation(op_name, arguments, manifest.oracle.parameters)
    outputs = [arg for arg in manifest.tensors if arg.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT)]
    if len(outputs) != 1:
        raise ValueError("builtin oracles currently require exactly one output")
    return {outputs[0].name: np.asarray(output, dtype=outputs[0].dtype)}


def validate_ir(kernel_ir: KernelIR, manifest: KernelManifest) -> IRValidationResult:
    manifest_inputs = {
        item.name: (item.dtype, item.shape)
        for item in manifest.tensors
        if item.direction != ArgumentDirection.OUTPUT
    }
    ir_inputs = {item.name: (item.dtype, item.shape) for item in kernel_ir.inputs}
    manifest_outputs = {
        item.name: (item.dtype, item.shape)
        for item in manifest.tensors
        if item.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT)
    }
    ir_outputs = {item.name: (item.dtype, item.shape) for item in kernel_ir.outputs}
    manifest_scalars = {item.name: (item.dtype, item.value) for item in manifest.scalars}
    ir_scalars = {item.name: (item.dtype, item.value) for item in kernel_ir.scalars}
    if (manifest_inputs, manifest_outputs, manifest_scalars) != (ir_inputs, ir_outputs, ir_scalars):
        return IRValidationResult(False, {}, "IR signature does not match the manifest")
    inputs = generate_inputs(manifest)
    expected = execute_oracle(manifest, inputs)
    actual = execute_ir(kernel_ir, inputs)
    mismatches: dict[str, float] = {}
    for name, expected_value in expected.items():
        if name not in actual:
            mismatches[name] = float("inf")
            continue
        actual_value = actual[name]
        if actual_value.shape != expected_value.shape:
            mismatches[name] = float("inf")
            continue
        if not np.allclose(
            actual_value,
            expected_value,
            rtol=manifest.tolerance.rtol,
            atol=manifest.tolerance.atol,
            equal_nan=manifest.tolerance.equal_nan,
        ):
            mismatches[name] = float(np.max(np.abs(actual_value.astype(np.float64) - expected_value.astype(np.float64))))
    passed = not mismatches
    return IRValidationResult(passed, mismatches, "IR matches oracle" if passed else f"IR mismatch: {mismatches}")
