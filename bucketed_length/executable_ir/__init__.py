"""Compare a descriptive JSON pivot with the existing executable KernelIR."""
from pydantic import BaseModel, ConfigDict, Field

from npu_agent.ir import validate_ir
from npu_agent.scenarios import ExecutableKernelIR
from npu_agent.study_v4 import IR_CONTRACT


class JsonSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    operation: str = Field(min_length=1)
    indexing: str = Field(min_length=1)
    numeric_behavior: str = Field(min_length=1)


def lift(call, manifest, context, mode, feedback=''):
    if mode == 'json_summary':
        value = call('Summarize the source semantics, indexing and numeric behavior as JSON.\n' + context + feedback,
                     JsonSummary)
        return value.model_dump(mode='json'), {'status': 'not_requested', 'executable': False}
    value = call(IR_CONTRACT + context + feedback, ExecutableKernelIR)
    result = validate_ir(value, manifest)
    return value.model_dump(mode='json'), {'status': 'passed' if result.passed else 'failed',
        'executable': True, 'message': result.message, 'reference': 'independent_builtin',
        'source_equivalence': 'not_established'}
