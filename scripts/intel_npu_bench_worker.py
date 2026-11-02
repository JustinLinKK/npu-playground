"""Long-lived Intel NPU benchmark worker for research/benchmark_kernel_energy.py (Windows-native Python).

One JSON command per stdin line, one JSON reply per line. The blob stays imported between commands, so an idle
worker is the NPU's loaded-but-idle baseline. Inputs are bound once into plugin-owned buffers (the first case, as in
the executor's timing), so timed calls exclude input copies. Method: docs/time-energy-measurement.md.

  load    {request, blob}        check the request and import its blob on the NPU alone (scripts/intel_npu_executor.py)
  build   {config, inputs, out, properties}
                                 compile a research/sweep_kernels.py configuration on the NPU alone, bind the inputs
                                 saved in <inputs> (.npz) and save one inference's output as <out> (.npz)
  cases   {out}                  run every request case and save the outputs as <out>/<case>.npz
  warmup  {count, seconds}       synchronous inferences, untimed; stops early after `seconds` if given
  rounds  {count, calls}         per-call times of synchronous inferences
  window  {mode, warm, seconds, tail, nireq}
                                 fixed window of back-to-back inferences, mode 'sync' or 'async'; prints
                                 {"event": "begin"} and {"event": "end", ...} the moment they happen, then
                                 {"event": "done", ...}
  bursts  {mode, calls, period, count, nireq}
                                 intermittent load: every `period` seconds a burst of `calls` inferences, `count`
                                 times, starting at once; prints {"event": "begin"} at each burst start and
                                 {"event": "end"} when the last period ends, then {"event": "done", ...}
  quit
"""
from __future__ import annotations

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research')]

import energy_meter  # noqa: E402
from intel_npu_executor import execution_devices, load_blob, npu_core  # noqa: E402  (also puts src/ on sys.path)
from npu_agent.executors import decode_tensor  # noqa: E402
from npu_agent.models import ExecutionRequest  # noqa: E402

COMMANDS = ('load', 'build', 'cases', 'warmup', 'rounds', 'window', 'bursts')


class Worker:
    def __init__(self, emit, clock=time.perf_counter):
        self.emit, self.clock, self.queues = emit, clock, {}

    def load(self, request: str, blob: str) -> dict:
        task = ExecutionRequest.model_validate_json(Path(request).read_text())
        # The request names the blob by its WSL path; the caller passes the same file's Windows path.
        task = task.model_copy(update={'artifacts': {'compiled.blob': blob}})
        self.ov, core, self.compiled, self.inputs, self.outputs, architecture, devices = load_blob(task)
        self.request = task
        first = next(iter(task.cases.values()))
        self.feed = {name: np.array(decode_tensor(first[name])) for name in self.inputs}
        self.infer = self.compiled.create_infer_request()
        self.infer.infer(self.feed)
        return {'device': str(core.get_property('NPU', 'FULL_DEVICE_NAME')), 'architecture': architecture,
                'execution_devices': devices, 'driver': str(core.get_property('NPU', 'NPU_DRIVER_VERSION')),
                'openvino': self.ov.get_version(),
                'optimal_nireq': int(self.compiled.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS'))}

    def build(self, config: str, inputs: str, out: str, properties: dict | None = None) -> dict:
        import sweep_kernels

        spec = sweep_kernels.SWEEP[config]
        self.ov, core = npu_core()
        started = self.clock()
        self.compiled = core.compile_model(sweep_kernels.openvino_model(spec), 'NPU', properties or {})
        compile_s = self.clock() - started
        devices = execution_devices(self.compiled)
        if devices != ['NPU']:
            raise RuntimeError(f'compiled model would run on {devices}, not the NPU alone')
        with np.load(inputs) as saved:
            self.feed = {name: saved[name] for name in spec['inputs']}
        self.infer, self.queues = self.compiled.create_infer_request(), {}
        self.infer.infer(self.feed)
        np.savez(out, out=np.array(self.infer.get_output_tensor(0).data, copy=True))
        return {'device': str(core.get_property('NPU', 'FULL_DEVICE_NAME')),
                'architecture': str(core.get_property('NPU', 'DEVICE_ARCHITECTURE')), 'execution_devices': devices,
                'driver': str(core.get_property('NPU', 'NPU_DRIVER_VERSION')), 'openvino': self.ov.get_version(),
                'compile_s': compile_s, 'properties': properties or {},
                'optimal_nireq': int(self.compiled.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS'))}

    def cases(self, out: str) -> dict:
        folder = Path(out)
        folder.mkdir(parents=True, exist_ok=True)
        for case_id, encoded in self.request.cases.items():
            self.infer.infer({name: np.array(decode_tensor(encoded[name])) for name in self.inputs})
            np.savez(folder / f'{case_id}.npz',
                     **{name: np.array(self.infer.get_tensor(name).data, copy=True) for name in self.outputs})
        self.infer.infer(self.feed)  # rebind the timed case
        return {'cases': len(self.request.cases)}

    def warmup(self, count: int, seconds: float | None = None) -> dict:
        deadline, done = (self.clock() + seconds) if seconds else None, 0
        while done < count and (deadline is None or self.clock() < deadline):
            self.infer.infer()
            done += 1
        return {'warmup': done}

    def rounds(self, count: int, calls: int) -> dict:
        return energy_meter.rounds(self.infer.infer, count, calls, self.clock)

    def step(self, mode: str, nireq: int):
        if mode == 'sync':
            return self.infer.infer, None
        if mode != 'async':
            raise ValueError(f'unknown mode: {mode}')
        if nireq not in self.queues:
            queue = self.ov.AsyncInferQueue(self.compiled, nireq)
            for index in range(len(queue)):
                queue[index].infer(self.feed)  # bind inputs; start_async() without inputs reuses them
            self.queues[nireq] = queue
        queue = self.queues[nireq]
        return queue.start_async, queue.wait_all

    def window(self, mode: str, warm: float, seconds: float, tail: float, nireq: int = 4) -> dict:
        step, finish = self.step(mode, nireq)
        result = energy_meter.run_window(
            step, warm, seconds, tail, self.clock, finish=finish,
            begin=lambda: self.emit({'event': 'begin', 't': self.clock()}),
            end=lambda window: self.emit({'event': 'end', 't': self.clock(), **window}))
        return {'event': 'done', 'mode': mode, 'nireq': nireq if mode == 'async' else 1, **result}


    def bursts(self, mode: str, calls: int, period: float, count: int, nireq: int = 4) -> dict:
        step, finish = self.step(mode, nireq)
        result = energy_meter.bursts(step, calls, period, count, self.clock, time.sleep, finish=finish,
                                     begin=lambda: self.emit({'event': 'begin', 't': self.clock()}),
                                     end=lambda: self.emit({'event': 'end', 't': self.clock()}))
        return {'event': 'done', 'mode': mode, **result}


def serve(worker: Worker, lines, emit) -> None:
    for line in lines:
        if not line.strip():
            continue
        command = json.loads(line)
        name = command.pop('cmd', None)
        if name == 'quit':
            emit({'ok': True})
            return
        try:
            if name not in COMMANDS:
                raise ValueError(f'unknown command: {name}')
            emit(getattr(worker, name)(**command))
        except Exception as exc:  # report to the orchestrator instead of dying with an unread traceback
            emit({'error': f'{type(exc).__name__}: {exc}'})


def main() -> int:
    out = sys.stdout.buffer  # bytes: Windows text streams would translate line endings

    def emit(reply):
        out.write((json.dumps(reply) + '\n').encode())
        out.flush()

    with redirect_stdout(sys.stderr):  # keep library prints off the protocol stream
        serve(Worker(emit), iter(sys.stdin.buffer.readline, b''), emit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
