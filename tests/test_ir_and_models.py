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
                index_expression="logical tensor indexing",
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

