from __future__ import annotations

import json
from enum import Enum, IntEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
MAX_GENERATED_FILE_BYTES = 1_000_000
_HARDWARE_TOKENS = (
    "threadidx",
    "blockidx",
    "blockdim",
    "griddim",
    "warp",
    "wavefront",
    "shared_memory",
    "__shared__",
    "objectfifo",
    "aie.tile",
    "npu4000",
    "npu2",
)


class Dialect(str, Enum):
    CUDA = "cuda"
    TRITON = "triton"
    HIP = "hip"


class ArgumentDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class Backend(str, Enum):
    AMD_XDNA2 = "amd_xdna2"
    INTEL_OPENVINO = "intel_openvino"


class EvidenceTier(IntEnum):
    INVALID = 0
    OFFLINE_COMPILE = 1
    HOST_EQUIVALENCE = 2
    HARDWARE_MEASURED = 3


class NumericTolerance(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rtol: float = Field(default=1e-5, ge=0)
    atol: float = Field(default=1e-6, ge=0)
    equal_nan: bool = False


class TensorArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    direction: ArgumentDirection
    dtype: Literal["float16", "float32", "float64", "int8", "int16", "int32", "int64"]
    shape: tuple[int, ...]
    layout: str = "row_major"
    strides: tuple[int, ...] | None = None
    alias_of: str | None = None

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("tensor shapes must be non-empty and fully static")
        return shape

    @model_validator(mode="after")
    def validate_strides(self) -> TensorArgument:
        if self.strides is not None and len(self.strides) != len(self.shape):
            raise ValueError("strides must have the same rank as shape")
        return self


class ScalarArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    dtype: Literal["float32", "float64", "int32", "int64", "bool"]
    value: int | float | bool


class OracleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["builtin"] = "builtin"
    operation: Literal[
        "vector_add",
        "reduce_sum",
        "matmul",
        "transpose",
        "softmax",
        "layer_norm",
        "attention",
        "moving_average",
        "conv2d",
        "gamma_correction",
        "scalar_mul",
        "sigmoid",
        "silu",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class KernelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source_file: str
    dialect: Dialect
    entrypoint: str = Field(min_length=1)
    tensors: list[TensorArgument] = Field(min_length=1)
    scalars: list[ScalarArgument] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tolerance: NumericTolerance = Field(default_factory=NumericTolerance)
    seed: int = 0
    oracle: OracleSpec
    provenance: dict[str, str] = Field(default_factory=dict)
    numerical_domain: Literal["finite"] = "finite"

    @model_validator(mode="after")
    def validate_arguments(self) -> KernelManifest:
        names = [arg.name for arg in self.tensors] + [arg.name for arg in self.scalars]
        if len(names) != len(set(names)):
            raise ValueError("argument names must be unique")
        if not any(arg.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT) for arg in self.tensors):
            raise ValueError("at least one output or inout tensor is required")
        outputs = [arg for arg in self.tensors if arg.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT)]
        if self.oracle.kind == "builtin" and len(outputs) != 1:
            raise ValueError("builtin oracles require exactly one output or inout tensor")
        known = set(names)
        for tensor in self.tensors:
            if tensor.alias_of is not None and tensor.alias_of not in known:
                raise ValueError(f"unknown alias target: {tensor.alias_of}")
        return self


class IterationDomain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variables: list[str]
    bounds: list[str]


class ReductionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axes: list[int]
    operator: Literal["sum", "max", "min", "product"]
    initial_value: int | float
    accumulation_dtype: str
    ordered: bool = False


class OperationAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis: int | None = None
    axes: list[int] | None = None
    keepdims: bool = False
    output_dtype: Literal["float16", "float32", "float64", "int8", "int16", "int32", "int64"] | None = None
    epsilon: float | None = None
    scale: float | None = None
    window: int | None = None
    padding: int | None = None
    gamma: float | None = None
    value: int | float | bool | None = None
    shape: list[int] | None = None
    accumulation_dtype: Literal["float32", "float64", "int32", "int64"] | None = None
    comparison: Literal["lt", "le", "eq", "ge", "gt", "ne"] | None = None


class IROperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    op: Literal[
        "add",
        "reduce_sum",
        "matmul",
        "transpose",
        "softmax",
        "layer_norm",
        "attention",
        "moving_average",
        "conv2d",
        "gamma_correction",
        "constant",
        "subtract",
        "multiply",
        "divide",
        "exp",
        "sqrt",
        "center",
        "maximum",
        "minimum",
        "compare",
        "select",
        "reshape",
        "cast",
        "reduce_max",
        "reduce_min",
        "reduce_product",
    ]
    inputs: list[str]
    output: str
    attributes: OperationAttributes = Field(default_factory=OperationAttributes)
    index_expression: str | None = None
    reduction: ReductionSpec | None = None


class KernelIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "2.0"] = SCHEMA_VERSION
    name: str
    inputs: list[TensorArgument]
    scalars: list[ScalarArgument] = Field(default_factory=list)
    outputs: list[TensorArgument]
    iteration_domains: list[IterationDomain]
    operations: list[IROperation] = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    numeric_behavior: list[str] = Field(default_factory=list)
    source_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantics(self) -> KernelIR:
        payload = self.model_dump(exclude={"source_evidence"})
        text = json.dumps(payload, sort_keys=True).lower()
        token = next((item for item in _HARDWARE_TOKENS if item in text), None)
        if token:
            raise ValueError(f"hardware-specific token is not permitted in semantic IR: {token}")
        values = {arg.name for arg in self.inputs} | {arg.name for arg in self.scalars}
        operation_ids: set[str] = set()
        for operation in self.operations:
            if operation.id in operation_ids:
                raise ValueError(f"duplicate operation id: {operation.id}")
            missing = set(operation.inputs) - values
            if missing:
                raise ValueError(f"operation {operation.id} references unknown inputs: {sorted(missing)}")
            operation_ids.add(operation.id)
            values.add(operation.output)
        missing_outputs = {arg.name for arg in self.outputs} - values
        if missing_outputs:
            raise ValueError(f"IR does not produce outputs: {sorted(missing_outputs)}")
        return self


class GeneratedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    content: str

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not value or ".." in path.parts or "." in path.parts:
            raise ValueError("generated paths must be normalized relative paths")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_GENERATED_FILE_BYTES:
            raise ValueError("generated file exceeds size limit")
        return value


class CodeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Backend
    files: list[GeneratedFile]
    unsupported_operations: list[str] = Field(default_factory=list)


class OptimizationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    rationale: str
    expected_benefit: str
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    bundle: CodeBundle


class ProposalSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[OptimizationProposal]


class DebugResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str
    fix_summary: str
    bundle: CodeBundle


class TargetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    vendor: Literal["amd", "intel"]
    backend: Backend
    hardware: str
    compiler_image: str
    compiler_version: str
    compiler_properties: dict[str, str] = Field(default_factory=dict)
    hardware_runner: list[str] | None = None
    executor_warmup_count: int = Field(default=0, ge=0)
    executor_measurement_count: int = Field(default=0, ge=0)


class ValidationPolicy(str, Enum):
    COMPILE_ONLY = "compile-only"
    OFFLINE_VALIDATED = "offline-validated"
    TARGET_EXECUTED = "target-executed"


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: Literal["passed", "failed", "blocked", "unsupported", "not_requested"]
    engine: str = ""
    representation: str = ""
    correct: bool | None = None
    reason_code: str | None = None
    error_category: str | None = None
    message: str = ""
    cases_run: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    artifacts: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> StageResult:
        if self.status in ("blocked", "unsupported", "not_requested") and self.correct is not None:
            raise ValueError("unexecuted stages must have unknown correctness")
        if self.status == "passed" and self.correct is False:
            raise ValueError("a passed stage cannot have failed correctness")
        return self


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["2.0"] = "2.0"
    candidate_id: str
    target_id: str
    source_sha256: str | None = None
    manifest_sha256: str | None = None
    input_sha256: dict[str, str] = Field(default_factory=dict)
    toolchain_fingerprint: str | None = None
    reference_origin: Literal["independent_builtin", "golden_fixture", "source_runtime"] | None = None
    stages: dict[str, StageResult] = Field(default_factory=dict)
    legacy: bool = False
    policy: ValidationPolicy = ValidationPolicy.COMPILE_ONLY
    policy_met: bool = False
    offline_contract_met: bool = False
    all_three_requirements_met: bool = False
    hardware_latency_p50_ms: float | None = Field(default=None, ge=0)
    estimated_latency_ms: float | None = Field(default=None, ge=0)


class TensorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str
    shape: list[int]
    data_base64: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    target: TargetProfile
    artifacts: dict[str, str]
    artifact_sha256: dict[str, str]
    manifest: KernelManifest
    manifest_sha256: str
    abi_sha256: str
    cases: dict[str, dict[str, TensorPayload]]
    compiler_fingerprint: str
    timeout_seconds: int = Field(gt=0)
    warmup_count: int = Field(default=0, ge=0)
    measurement_count: int = Field(default=0, ge=0)
    timing_scope: Literal["execution", "end_to_end"] = "execution"


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    target_id: str
    hardware: str
    executor_kind: Literal["binary_simulator", "physical_npu"]
    executed_artifact_sha256: dict[str, str]
    outputs: dict[str, dict[str, TensorPayload]]
    logs: str
    runtime_identity: str = Field(min_length=1)
    compatibility_checked: Literal[True]
    timing_samples_ms: list[float] = Field(default_factory=list)
    warmup_count: int = Field(default=0, ge=0)
    measurement_count: int = Field(default=0, ge=0)
    timing_scope: Literal["execution", "end_to_end"] = "execution"
    timing_provenance: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> ExecutionResult:
        if len(self.timing_samples_ms) != self.measurement_count:
            raise ValueError("timing sample count mismatch")
        if any(value < 0 for value in self.timing_samples_ms):
            raise ValueError("latency cannot be negative")
        if self.timing_samples_ms and not self.timing_provenance:
            raise ValueError("timing requires provenance")
        if not self.executed_artifact_sha256:
            raise ValueError("execution requires artifact identities")
        return self


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    compiler_fingerprint: str
    duration_seconds: float = Field(default=0.0, ge=0)
    evaluation_duration_seconds: float = Field(default=0.0, ge=0)
    host_correct: bool | None = None
    hardware_correct: bool | None = None
    cpu_latency_p50_ms: float | None = None
    cpu_latency_p95_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    hardware_warmup_count: int | None = None
    hardware_iteration_count: int | None = None
    hardware_throughput_per_second: float | None = None
    static_metrics: dict[str, float] = Field(default_factory=dict)
    validation: ValidationResult | None = None


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    evidence_tier: EvidenceTier
    compile_success: bool
    correctness: bool | None
    host_correctness: bool | None = None
    target_correctness: bool | None = None
    reward: float = Field(ge=0, le=1)
    static_score: float = Field(ge=0, le=1)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cpu_latency_p50_ms: float | None = None
    cpu_latency_p95_ms: float | None = None
    artifact_bytes: int = 0
    duration_seconds: float = Field(default=0.0, ge=0)
    notes: list[str] = Field(default_factory=list)
    validation: ValidationResult | None = None


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    target_id: str
    parent_id: str | None = None
    depth: int = 0
    label: str
    rationale: str
    bundle: CodeBundle
    evaluation: Evaluation | None = None
    debug_attempt: int = 0


class TargetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: TargetProfile
    winner: Candidate | None
    candidates_evaluated: int
    status: Literal["completed", "failed", "blocked"]
    error: str | None = None


class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    manifest_path: str
    targets: list[TargetProfile] = Field(min_length=1)
    provider: Literal["openai", "codex-cli", "claude-cli", "openrouter"] = "openai"
    model: str | None = None
    search_rounds: int = Field(default=3, ge=0, le=20)
    branching_factor: int = Field(default=3, ge=1, le=8)
    debug_retries: int = Field(default=2, ge=0, le=5)
    validation_policy: ValidationPolicy = ValidationPolicy.COMPILE_ONLY


class TranslationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    kernel: str
    ir: KernelIR
    ir_validated: bool
    targets: list[TargetResult]
    status: Literal["completed", "partial", "failed", "blocked"]
    report_path: str
    duration_seconds: float = Field(default=0.0, ge=0)


class BaselineTargetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: TargetProfile
    candidate: Candidate | None = None
    status: Literal["completed", "failed", "blocked"]
    error: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0)


class BaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    kernel: str
    targets: list[BaselineTargetResult]
    status: Literal["completed", "partial", "failed", "blocked"]
    report_path: str
    duration_seconds: float = Field(default=0.0, ge=0)
