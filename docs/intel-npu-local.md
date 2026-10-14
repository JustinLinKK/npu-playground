# Intel NPU 3720 on a local machine (WSL2)

This guide runs exported Intel NPU blobs on a physical Intel NPU and records the
outputs as `target_execution` evidence. The offline container still builds each
OpenVINO graph, runs it on the CPU plugin and compiles the NPU blob. Only the final
step, loading that exact blob and running the twelve input cases, uses the device.

## Which target

Use `intel_npu_3720` for Meteor Lake and Arrow Lake NPUs, for example a Core
Ultra 200-series CPU whose NPU reports PCI ID `8086:AD1D` and OpenVINO
`DEVICE_ARCHITECTURE` `3720`.

The historical `intel_npu_4000` target compiles for Lunar Lake. Its blobs do not
load on an NPU 3720: the driver rejects them with
`ZE_RESULT_ERROR_INVALID_NATIVE_BINARY`. The two targets share the same compiler
image and differ only in `NPU_PLATFORM`.

The CLI and three-scenario experiment defaults remain AMD XDNA2 plus Intel NPU
4000. Select the 3720 target explicitly with `--targets intel_npu_3720`.

## Why WSL needs a Windows bridge

WSL2 kernels are built without `CONFIG_DRM_ACCEL`. They have no `/dev/accel`
device, so the Linux OpenVINO NPU plugin and the containers cannot see the NPU.
Passing a device to Docker cannot change this.

Windows-native OpenVINO can see the NPU, and WSL can start Windows programs:

1. `scripts/wsl_npu_bridge.py` is the target's `hardware_runner`. It runs in WSL,
   rewrites artifact paths with `wslpath -w`, and holds `.npu-agent/npu.lock` so
   that parallel case workers use the NPU one at a time.
2. `scripts/intel_npu_executor.py` runs in Windows Python with `-B` and reads the
   repository through `\\wsl.localhost\...`. It checks the request, blob hash and
   device architecture. It then imports the blob on the device named `NPU`, never
   AUTO or HETERO, and requires `EXECUTION_DEVICES` to be exactly `NPU`.
3. The supervisor compares the returned tensors with the independent oracle.

## Windows setup

Install Windows Python (3.12 or newer) and create a virtual environment with the
same OpenVINO release as the compiler image. NPU blobs are not portable across
OpenVINO versions.

```powershell
py -m venv C:\npu-runner\venv
C:\npu-runner\venv\Scripts\pip install openvino==2026.3.1 numpy==2.5.2 pydantic==2.12.5
```

Tested configuration (October 13, 2026):

| Component | Version |
| --- | --- |
| Windows Python | 3.14.4 |
| OpenVINO runtime | `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, identical in the compiler image |
| NPU | Intel AI Boost, architecture `3720`, driver 32.0.100.4404 (`NPU_DRIVER_VERSION` 1004404) |

Check the device from WSL. The probe compiles a small add graph whose float16
result is exact and compares it bit for bit:

```bash
.venv/bin/python scripts/wsl_npu_bridge.py \
  --windows-python /mnt/c/npu-runner/venv/Scripts/python.exe --probe
```

## Enable the runner

```bash
export NPU_AGENT_INTEL_NPU_3720_RUNNER="$PWD/.venv/bin/python $PWD/scripts/wsl_npu_bridge.py --windows-python /mnt/c/npu-runner/venv/Scripts/python.exe"
```

- **Scope.** `NPU_AGENT_<TARGET_ID>_RUNNER` sets `hardware_runner` for that
  profile only. When it is unset, profiles are unchanged and `target-executed`
  stays blocked (exit 3).
- **Timing.** `NPU_AGENT_<TARGET_ID>_TIMING="warmup,measure"` requests timing
  samples, measured with `perf_counter` around synchronous inference of the first
  case. Without it, runs check correctness only.
- **Fingerprints.** The runner is excluded from the toolchain fingerprint but
  included in campaign identity. Keep the same value for a campaign and all of its
  resumes.

## Hardware validation without model calls

```bash
uv run npu-agent env smoke --targets intel_npu_3720
uv run npu-agent validate-suite tests/fixtures/backends --targets intel_npu_3720 \
  --validation-policy target-executed --report-dir runs/npu-hw-validation/fixtures
uv run python scripts/prepare_validation_corpus.py
uv run npu-agent validate-suite runs/validation-corpus --targets intel_npu_3720 \
  --validation-policy target-executed --report-dir runs/npu-hw-validation/corpus
```

Judge the device by execution evidence, not by exit code. Every
`target_execution` stage should report:

- `engine: physical_npu`;
- `cases_run: 12`;
- status `passed`, or `failed` with `error_category: numerical_mismatch`.

`execution_process`, `malformed_report` and `timeout` indicate executor or device
problems.

**The NPU computes in FP16.** It reports no float32 throughput, and its
inference precision is f16. Most manifests use float32 tolerances such as
`rtol=1e-5, atol=1e-6`, so many correct programs report numerical mismatches on
the NPU. Every comparison records `max_absolute_error`.

Results from the tested configuration:

| Set | Executed on NPU | Within manifest tolerance | Numerical mismatch | Execution errors |
| --- | ---: | ---: | ---: | ---: |
| Fixtures (`add`, `reduction`, `sigmoid`) | 3 (36 cases) | 1 | 2 | 0 |
| Validation corpus (11 kernels) | 11 (132 cases) | 3 | 8 | 0 |

**The mismatches are FP16 representation effects.** Every case that failed the
float32 tolerance had an error consistent with FP16 rounding. For example:

- `add` `normal-*` cases had a maximum error of 6.5e-4 to 8.8e-4. This is the
  same as rounding the inputs to FP16.
- Transpose's `operation_boundary` input reaches 6143, where the FP16 spacing is
  2.0. Its maximum error was exactly 2.0.
- The sum 3836.625 rounds to 3836 in FP16.

## NPU outputs and the FP16 verdict

Every NPU execution now saves the device's actual output tensors as
`target_execution/<case>.npz`, with hashes recorded in the stage artifacts. Later
verdicts therefore never need the device again.

The supervisor records two verdicts:

- **Manifest verdict.** The stage's `correct` field and the `target-executed`
  policy still use the manifest's float32 tolerance against the independent
  oracle. `cases_passed` counts the cases that pass.
- **FP16 verdict.** `fp16_correct`, `fp16_cases_passed` and `fp16_comparisons`
  model what an exact FP16 device returns: round every float input and float
  scalar to FP16, evaluate the independent oracle on those values, and round the
  result to FP16. The NPU output passes if every element is within 8 FP16
  half-ulps (8 × 2⁻¹¹) of that output tensor's peak magnitude, plus 2⁻²⁴ for
  all-zero references. Each comparison also records `max_error_half_ulps`.

**Why model both roundings.** The device really does receive FP16 inputs. If
the reference rounded only the output, input rounding would count against the
kernel, and ill-conditioned operations amplify it. For example, layer-norm rows of
1 ± 8e-7 become exactly 1.0 in FP16, so a correct device returns the bias.

**What remains.** The 8-half-ulp allowance covers rounding inside the device:
intermediates, accumulation order and approximate transcendentals. The verdict
does not emulate each candidate operation in FP16. That would tie the reference
to the candidate's own algorithm.

**Known limitation.** The allowance scales with the output's peak magnitude. An
output that is near zero everywhere because terms cancel, like the matmul
`cancellation` case, can fail although the error is far below the magnitude of
the summed products.

Re-run a finished campaign's compiled blobs, with no model calls, to obtain both
verdicts for campaigns recorded earlier:

```bash
.venv/bin/python scripts/replay_intel_local_npu.py RUN_ROOT/evaluation --output RUN_ROOT/npu-replay-fp16
```

The replay checks each blob against its compile-time hash and saves outputs, rows
and `summary.json` in the new directory. It never rewrites the campaign.

On `intel-npu-baseline-openai-20261013T205319939819544Z` the replay produced:

- **Executions:** 49 attempts with no execution errors. All 49 reproduced their
  recorded manifest case counts.
- **Accepted tasks:** of 39, 2 met the manifest tolerance and 38 passed the FP16
  verdict. The one FP16 failure is the matmul `cancellation` case described above.
- **Case pass rates:** over 468 accepted-task cases, 38.7% under the manifest
  tolerance and 99.8% under the FP16 verdict.

## Direct OpenAI Intel baseline

`intel-npu-baseline-v1` mirrors `xdna2-baseline-v1`: the forty-task corpus, one
direct arm, one repetition, the same calibration gate and the same launcher logic.
It runs on `intel_npu_3720` with the unguided OpenVINO contract used by the v4
direct arm.

`configs/intel-npu-baseline-openai.json` matches
`configs/xdna2-baseline-openai.json` except for protocol and target. It uses
`gpt-6-astra` at medium reasoning with no temperature or top-p. Put the key in
`openai.api_key` in the ignored `config.yaml`, as described in the
[XDNA2 baseline guide](xdna2-baseline.md).

```bash
export NPU_AGENT_INTEL_NPU_3720_RUNNER="..."               # required; the launcher refuses to start without it
LC_ALL=C.utf8 bash scripts/run_intel_npu_baseline_openai.sh             # prints RUN_ROOT
LC_ALL=C.utf8 bash scripts/run_intel_npu_baseline_openai.sh RUN_ROOT    # resume with the same runner value
uv run python research/run_xdna2_baseline.py report --root RUN_ROOT
```

Physical execution is observational:

- Acceptance and repair feedback remain `offline-validated`, which means target
  compilation plus OpenVINO CPU correctness.
- The repair prompt never receives `target_execution`.
- If a runner is configured, preflight requires the smoke fixture to have
  executed on the physical NPU before any model call.
- `results/REPORT.md` adds a "Physical NPU observation" section with case pass rates
  under both verdicts. The per-task and per-attempt CSV files add NPU status, case
  counts, maximum absolute error and the FP16 verdict.
- Like the AMD launcher, this one has no local dollar cap. Check billing on
  OpenAI.
