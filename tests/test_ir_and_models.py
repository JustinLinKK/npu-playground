from pathlib import Path

import pytest

from npu_agent.ir import validate_ir
from npu_agent.models import (
    ArgumentDirection,
    IROperation,
    IterationDomain,
    KernelIR,
    KernelManifest,
)


@pytest.mark.parametrize("axis", [0, -1])
def test_center_preserves_small_variations_near_a_common_offset(axis):
    import numpy as np
    from npu_agent.ir import evaluate_operation

    values = (1 + (np.arange(512, dtype=np.float32).reshape(4, 128) % 3) * np.float32(1e-6))
    if axis == 0:
        values = values.T
    reference = values.astype(np.float64) - values.astype(np.float64).mean(axis=axis, keepdims=True)
    ordinary = values - values.mean(axis=axis, keepdims=True, dtype=np.float32)
    stable = evaluate_operation("center", [values], {"axis": axis, "accumulation_dtype": "float32"})
    assert stable.dtype == np.float32
    assert np.max(np.abs(stable - reference)) < np.max(np.abs(ordinary - reference)) / 1000
    with pytest.raises(ValueError, match="one axis"):
        evaluate_operation("center", [values], {"axes": [axis]})


def test_center_based_layer_norm_ir_matches_all_independent_cases():
    manifest = KernelManifest.model_validate_json(Path("examples/classic/06_triton_layer_norm/manifest.json").read_text())
    ir = ir_for_manifest(manifest)
    ir.schema_version = "2.0"
    steps = [
        ("center", ["input"], "centered", {"axis": -1, "accumulation_dtype": "float32"}),
        ("multiply", ["centered", "centered"], "square", {}),
        ("reduce_sum", ["square"], "sum", {"axis": -1, "keepdims": True, "accumulation_dtype": "float32"}),
        ("constant", [], "width", {"value": 128, "output_dtype": "float32"}),
        ("divide", ["sum", "width"], "variance", {}),
        ("constant", [], "epsilon", {"value": 1e-5, "output_dtype": "float32"}),
        ("add", ["variance", "epsilon"], "regularized", {}),
        ("sqrt", ["regularized"], "denominator", {}),
        ("divide", ["centered", "denominator"], "normalized", {}),
        ("multiply", ["normalized", "weight"], "scaled", {}),
        ("add", ["scaled", "bias"], "output", {}),
    ]
    ir.operations = [IROperation(id=output, op=op, inputs=inputs, output=output, attributes=attributes)
                     for op, inputs, output, attributes in steps]
    result = validate_ir(ir, manifest)
    assert result.passed, result.message
    ir.schema_version = "1.0"
    assert "schema_version 2.0" in validate_ir(ir, manifest).message


def ir_for_manifest(manifest: KernelManifest) -> KernelIR:
    op_name = {"vector_add": "add"}.get(manifest.oracle.operation, manifest.oracle.operation)
    inputs = [tensor for tensor in manifest.tensors if tensor.direction != ArgumentDirection.OUTPUT]
    outputs = [
        tensor
        for tensor in manifest.tensors
        if tensor.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT)
    ]
    return KernelIR(
        name=manifest.name,
        inputs=inputs,
        scalars=manifest.scalars,
        outputs=outputs,
        iteration_domains=[IterationDomain(variables=["i"], bounds=["0 <= i < output_elements"])],
        operations=[
            IROperation(
                id="result",
                op=op_name,
                inputs=[tensor.name for tensor in inputs],
                output=outputs[0].name,
                attributes=manifest.oracle.parameters,
            )
        ],
        numeric_behavior=["use the manifest tolerance contract"],
        source_evidence=["threadIdx and ObjectFifo text are allowed only as provenance"],
    )


def test_all_classic_manifests_have_executable_reference_semantics() -> None:
    paths = sorted(Path("examples/classic").glob("*/manifest.json"))
    assert len(paths) == 10
    for path in paths:
        manifest = KernelManifest.model_validate_json(path.read_text(encoding="utf-8"))
        result = validate_ir(ir_for_manifest(manifest), manifest)
        assert result.passed, f"{manifest.name}: {result.message}"


def test_hardware_concepts_are_rejected_from_semantic_ir() -> None:
    manifest = KernelManifest.model_validate_json(
        Path("examples/classic/01_cuda_vector_add/manifest.json").read_text(encoding="utf-8")
    )
    kernel_ir = ir_for_manifest(manifest)
    payload = kernel_ir.model_dump(mode="json")
    payload["numeric_behavior"] = ["uses threadIdx ordering"]
    with pytest.raises(ValueError, match="hardware-specific token"):
        KernelIR.model_validate(payload)


def test_dynamic_shapes_are_rejected() -> None:
    payload = KernelManifest.model_validate_json(
        Path("examples/classic/01_cuda_vector_add/manifest.json").read_text(encoding="utf-8")
    ).model_dump(mode="json")
    payload["tensors"][0]["shape"] = [-1]
    with pytest.raises(ValueError, match="fully static"):
        KernelManifest.model_validate(payload)



@pytest.mark.parametrize('operation', ['sigmoid', 'silu'])
def test_stable_primitive_composition(operation):
    payload = KernelManifest.model_validate_json(
        Path('tests/fixtures/backends/intel_npu_4000/sigmoid/manifest.json').read_text()).model_dump(mode='json')
    payload['oracle']['operation'] = operation
    manifest = KernelManifest.model_validate(payload)
    ops = [
        IROperation(id='zero', op='constant', inputs=[], output='zero', attributes={'value': 0, 'output_dtype': 'float32'}),
        IROperation(id='one', op='constant', inputs=[], output='one', attributes={'value': 1, 'output_dtype': 'float32'}),
        IROperation(id='neg', op='subtract', inputs=['zero', 'x'], output='neg'),
        IROperation(id='abs', op='maximum', inputs=['x', 'neg'], output='abs'),
        IROperation(id='negabs', op='subtract', inputs=['zero', 'abs'], output='negabs'),
        IROperation(id='e', op='exp', inputs=['negabs'], output='e'),
        IROperation(id='den', op='add', inputs=['one', 'e'], output='den'),
        IROperation(id='positive', op='divide', inputs=['one', 'den'], output='positive'),
        IROperation(id='negative', op='divide', inputs=['e', 'den'], output='negative'),
        IROperation(id='sign', op='compare', inputs=['x', 'zero'], output='sign', attributes={'comparison': 'ge'}),
        IROperation(id='sigmoid', op='select', inputs=['sign', 'positive', 'negative'], output='output' if operation == 'sigmoid' else 'sigmoid'),
    ]
    if operation == 'silu':
        ops.append(IROperation(id='silu', op='multiply', inputs=['x', 'sigmoid'], output='output'))
    kernel_ir = KernelIR(schema_version='2.0', name=operation, inputs=[manifest.tensors[0]],
                         outputs=[manifest.tensors[1]], iteration_domains=[], operations=ops)
    result = validate_ir(kernel_ir, manifest)
    assert result.passed, result.message
