from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import ArgumentDirection, KernelIR, KernelManifest
from .oracles import execute_oracle, validate_goldens
from .testcases import generate_cases
from .validation import check_contract, compare_outputs


@dataclass(frozen=True)
class IRValidationResult:
    passed: bool
    mismatches: dict[str, float]
    message: str


def generate_inputs(manifest: KernelManifest) -> dict[str, np.ndarray | int | float | bool]:
    check_contract(manifest)
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
    height = height + 2 * padding - kernel_h + 1
    width = width + 2 * padding - kernel_w + 1
    if min(height, width) <= 0:
        raise ValueError("invalid conv2d output dimensions")
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
    if operation == "constant":
        return np.asarray(attributes["value"], dtype=attributes["output_dtype"])
    if operation == "center":
        dtype = attributes.get("accumulation_dtype", "float32")
        if dtype not in ("float32", "float64") or attributes.get("axes") is not None:
            raise ValueError("center requires one axis and float32 or float64 accumulation_dtype")
        value = np.asarray(inputs[0], dtype=dtype)
        axis = int(attributes.get("axis", -1))
        shifted = value - np.take(value, [0], axis=axis)
        return shifted - shifted.mean(axis=axis, keepdims=True, dtype=dtype)
    binary = {"subtract": np.subtract, "multiply": np.multiply, "divide": np.divide,
              "maximum": np.maximum, "minimum": np.minimum}
    if operation in binary:
        return binary[operation](*inputs)
    if operation in ("exp", "sqrt"):
        return {"exp": np.exp, "sqrt": np.sqrt}[operation](inputs[0])
    if operation == "compare":
        return {"lt": np.less, "le": np.less_equal, "eq": np.equal, "ge": np.greater_equal,
                "gt": np.greater, "ne": np.not_equal}[attributes["comparison"]](*inputs)
    if operation == "select":
        return np.where(*inputs)
    if operation == "reshape":
        return np.reshape(inputs[0], attributes["shape"])
    if operation == "cast":
        return np.asarray(inputs[0]).astype(attributes["output_dtype"])
    if operation in ("reduce_max", "reduce_min", "reduce_product"):
        axis = tuple(attributes["axes"]) if attributes.get("axes") is not None else attributes.get("axis")
        kwargs = {"axis": axis, "keepdims": attributes.get("keepdims", False)}
        if "initial_value" in attributes:
            kwargs["initial"] = attributes["initial_value"]
        if operation == "reduce_product":
            kwargs["dtype"] = attributes["accumulation_dtype"]
        return {"reduce_max": np.max, "reduce_min": np.min, "reduce_product": np.prod}[operation](inputs[0], **kwargs)
    if operation == "add":
        return np.asarray(inputs[0]) + np.asarray(inputs[1])
    if operation == "reduce_sum":
        axis = tuple(attributes["axes"]) if attributes.get("axes") is not None else attributes.get("axis")
        return np.asarray(inputs[0]).sum(
            axis=axis,
            dtype=attributes.get("accumulation_dtype", "float32"),
            keepdims=bool(attributes.get("keepdims", False)),
            initial=attributes.get("initial_value", 0),
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
    if kernel_ir.side_effects:
        raise ValueError("unsupported IR side_effects")
    values = dict(inputs)
    for operation in kernel_ir.operations:
        legacy_ops = {"add", "reduce_sum", "matmul", "transpose", "softmax", "layer_norm", "attention", "moving_average", "conv2d", "gamma_correction"}
        if kernel_ir.schema_version == "1.0" and operation.op not in legacy_ops:
            raise ValueError("primitive operations require KernelIR schema_version 2.0")
        if operation.index_expression:
            raise ValueError("unsupported executable index_expression; use structured operations")
        if operation.reduction:
            reduction = operation.reduction
            axes = operation.attributes.axes
            if axes is None and operation.attributes.axis is not None:
                axes = [operation.attributes.axis]
            expected_op = {"sum": "reduce_sum", "max": "reduce_max", "min": "reduce_min", "product": "reduce_product"}[reduction.operator]
            if (reduction.ordered or operation.op != expected_op or axes != reduction.axes
                    or reduction.accumulation_dtype != (operation.attributes.accumulation_dtype or "float32")):
                raise ValueError("unsupported reduction metadata")
        args = [values[name] for name in operation.inputs]
        attributes = operation.attributes.model_dump(exclude_none=True)
        if operation.reduction:
            attributes["initial_value"] = operation.reduction.initial_value
        values[operation.output] = evaluate_operation(operation.op, args, attributes)
    return {output.name: np.asarray(values[output.name]) for output in kernel_ir.outputs}



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
    mismatches: dict[str, float] = {}
    try:
        validate_goldens(manifest.oracle.operation)
        for case in generate_cases(manifest):
            expected = execute_oracle(manifest, case.inputs)
            actual = execute_ir(kernel_ir, case.inputs)
            comparison = compare_outputs(actual, expected, manifest.tolerance, case.id)
            if not comparison["correct"]:
                return IRValidationResult(False, {case.id: 1.0}, f"IR mismatch: {comparison}")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return IRValidationResult(False, {}, str(exc))
    return IRValidationResult(True, mismatches, "IR matches independent oracle on 12 cases; prose metadata is not executable evidence")
