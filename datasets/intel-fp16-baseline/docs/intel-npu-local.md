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

The float32 mismatch results below describe historical campaigns. New Intel
baseline campaigns use the prepared FP16 contracts described in the next section.

Every NPU execution now saves the device's actual output tensors as
`target_execution/<case>.npz`, with hashes recorded in the stage artifacts. Later
verdicts therefore never need the device again.

The supervisor records two verdicts:

- **Manifest verdict.** The stage's `correct` field and the `target-executed`
  policy use the campaign manifest's dtype and tolerance against the independent
  oracle. `cases_passed` counts the cases that pass.
- **FP16 verdict.** `fp16_correct`, `fp16_cases_passed` and `fp16_comparisons`
  model what an exact FP16 device returns: round every float input and float
  scalar to FP16, evaluate the independent oracle on those values, and round the
  result to FP16. Each element must satisfy
  `|npu − reference| ≤ atol + rtol·|reference|` with `rtol = 1e-3, atol = 1e-5`.
  These are PyTorch's float16 defaults (`torch.testing.assert_close`). They are
  set in `src/npu_agent/executors.py`, and the baseline pipeline uses them
  automatically.

**Why model both roundings.** The device really does receive FP16 inputs. If
the reference rounded only the output, input rounding would count against the
kernel, and ill-conditioned operations amplify it. For example, layer-norm rows of
1 ± 8e-7 become exactly 1.0 in FP16, so a correct device returns the bias.
Rounding the reference's result to FP16 means a perfect FP16 device scores zero
error.

**What the tolerances imply.**

- `rtol = 1e-3` allows about one FP16 ulp relative to each element.
- `atol = 1e-5` is below FP16's resolution at unit magnitude, so two device
  effects count as failures:
  - **Subnormal flush-to-zero.** The NPU returns FP16 subnormals (below 6.1e-5)
    as exactly 0. Each comparison counts these in `flushed_subnormals`.
  - **Cancellation residue.** Elements that land near zero because large terms
    cancel keep an absolute error at the resolution of those terms.
- `rtol = atol = 2⁻¹⁰` (one FP16 ulp at magnitude 1) absorbs both effects. Use the
  rescore command below to compare settings.

Each comparison also records `max_scaled_error` (error divided by its bound) and
`max_error_half_ulps` (error relative to the tensor's peak magnitude).

**Replay.** Re-run a finished campaign's compiled blobs, with no model calls, to
obtain both verdicts for campaigns recorded earlier:

```bash
.venv/bin/python scripts/replay_intel_local_npu.py RUN_ROOT/evaluation --output RUN_ROOT/npu-replay-fp16
```

The replay checks each blob against its compile-time hash and saves outputs, rows
and `summary.json` in the new directory. It never rewrites the campaign.

**Rescore.** Rescore saved outputs under any tolerance, with neither the device
nor model calls. Add `--outputs RUN_ROOT/npu-replay-fp16` for campaigns recorded
before outputs were saved.

```bash
.venv/bin/python scripts/rescore_intel_npu_outputs.py RUN_ROOT/evaluation --rtol 9.765625e-4 --atol 9.765625e-4
```

**Results.** These are for accepted tasks only, 39 tasks with 468 cases. Both
October 13–14 campaigns gave identical numbers:

| Verdict | Tasks passing all cases | Case pass rate |
| --- | ---: | ---: |
| Manifest float32 tolerance | 2 | 38.7% |
| FP16, `rtol 1e-3, atol 1e-5` (default; PyTorch's float16 default) | 22 | 82.9% |
| FP16, `rtol = atol = 2⁻¹⁰` | 35 | 95.7% |

**Remaining failures at `rtol = atol = 2⁻¹⁰`.** The four tasks still failing
exceed their bound by 3–4.6×:

- **`softplus`.** The NPU approximates transcendental functions.
- **`cumsum`, `reverse_cumsum` and the HIP convolution.** Running sums reach
  magnitudes of 10–60, so near-zero elements carry errors above unit-scale
  resolution.

Across these tasks the device flushed 4,067 subnormal reference values to zero.

## Prepared FP16 baseline data

Generate the forty-task dataset without model calls, OpenVINO, or an NPU:

```bash
.venv/bin/python scripts/prepare_intel_fp16_corpus.py --output runs/intel-fp16-corpus
```

Each task contains twelve deterministic cases, for **480 input/reference pairs**:

- `manifest.json`: FP16 floating tensor inputs and outputs, with
  `rtol=1e-3, atol=1e-5`. Integer tensors and scalar types are preserved.
- `inputs/<case>.npz`: the actual FP16 input tensors, keyed by manifest names.
- `expected/<case>.npz`: independent reference outputs rounded to FP16.
- `reference-fp32/<case>.npz`: the reference outputs before final FP16 rounding,
  for numerical analysis.
- `original-manifest.json` and the original GPU source: provenance for the
  precision adaptation. The source still describes the original GPU interface;
  the prepared manifest explicitly takes precedence for dtypes and scalar values.
- `dataset.json` at the dataset root: source/generator hashes, NumPy version,
  case inventory, precision policy, and SHA-256 hashes for all exported files.

References are recomputed from the saved FP16 inputs using the existing
independent oracle with FP32 or higher intermediate calculations. They are never
obtained by casting outputs computed from the old FP32 inputs. Floating scalar
arguments are rounded to FP16 values, stored in their existing scalar type, and
their bound oracle parameters are updated to match. Other algorithm parameters,
including normalization epsilon, keep their specified values.

Load both NPZ files with `numpy.load(..., allow_pickle=False)` and compare by
tensor name using `npu_agent.validation.compare_outputs(actual, expected,
manifest.tolerance, case_id)`. It checks output names, shapes, dtypes, and the
elementwise bound `abs(actual - expected) <= atol + rtol * abs(expected)`.
For a legacy blob with FP32 ports, widen these exact FP16 inputs to FP32 at its
interface; compare output values in FP32 with the FP16 reference widened to FP32.
Do not regenerate FP32 inputs for that comparison.

The standard reference is not an emulation of every NPU intermediate rounding,
approximation, or subnormal flush. Those effects can still cause real differences
under the recorded tolerance. OpenVINO accepts F32 and F16 graph types while the
NPU hardware computes in FP16; see the
[OpenVINO NPU precision documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html).

Re-running the preparer verifies an existing dataset's identity and hashes.
Changed or incomplete datasets require a new output directory. The original
`examples/` manifests and historical campaigns are preserved. The existing CUDA
source-validation report applies to the original GPU contracts; this export
validates the adapted independent references, not FP16 CUDA execution or physical
NPU correctness.

## Direct OpenAI Intel baseline

`intel-npu-baseline-v1` mirrors `xdna2-baseline-v1`: the forty-task corpus, one
direct arm, one repetition, the same calibration gate and the same launcher logic.
It runs on `intel_npu_3720` with the unguided OpenVINO contract used by the v4
direct arm.

The launcher now prepares `RUN_ROOT/calibrate-fp16-inputs` and
`RUN_ROOT/evaluate-fp16-inputs` automatically and points calibration/evaluation
at those FP16 contracts. The same deterministic inputs and reference calculation
are used by the campaign and the exported dataset. Evaluation records the
dataset report hash in `evaluation/preparation.json`. Use a **fresh RUN_ROOT**:
an old FP32 calibration cannot qualify an FP16 evaluation, and old campaign
configurations cannot be silently resumed with a different precision contract.

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
- `results/REPORT.md` adds a second headline line: tasks accepted offline **and**
  verified on the physical NPU under the FP16 verdict, out of all planned tasks.
- Its "Physical NPU verification" section reports each rate twice. *Of planned*
  counts tasks or cases without an accepted candidate as failures. *Of accepted*
  covers accepted candidates only.
- The FP16 verdict is rescored from the hash-checked saved NPU outputs, using the
  current tolerances, every time the report is built. Regenerate it with
  `research/run_xdna2_baseline.py report --root RUN_ROOT`.
- The per-task and per-attempt CSV files add NPU status, case counts and both
  verdicts.
- Like the AMD launcher, this one has no local dollar cap. Check billing on
  OpenAI.
