import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_intel_npu_executor import Port, request


ROOT = Path(__file__).resolve().parents[1]


class Request:
    def __init__(self):
        self.bound, self.state, self.calls = None, {}, 0

    def infer(self, feed=None):
        if feed is not None:
            self.bound = {name: np.array(value) for name, value in feed.items()}
        self.calls += 1
        self.state['output'] = (self.bound['x'] + self.bound['y']).astype(np.float32)
        return self.state

    def get_tensor(self, name):
        return SimpleNamespace(data=self.state[name])


class Queue:
    def __init__(self, compiled, nireq):
        self.requests, self.started, self.waits = [Request() for _ in range(nireq)], 0, 0

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        return self.requests[index]

    def start_async(self):
        assert all(r.bound is not None for r in self.requests), 'inputs must be bound before start_async()'
        self.started += 1

    def wait_all(self):
        self.waits += 1


class Compiled:
    def __init__(self, manifest):
        self.inputs = [Port(t) for t in manifest.tensors if t.direction == 'input']
        self.outputs = [Port(t) for t in manifest.tensors if t.direction == 'output']

    def get_property(self, name):
        return {'EXECUTION_DEVICES': 'NPU', 'OPTIMAL_NUMBER_OF_INFER_REQUESTS': 1}[name]

    def create_infer_request(self):
        return Request()


@pytest.fixture
def worker_module(monkeypatch, tmp_path):
    task = request(tmp_path)

    class Core:
        available_devices = ['CPU', 'NPU']

        def get_property(self, device, name):
            return {'DEVICE_ARCHITECTURE': '3720', 'FULL_DEVICE_NAME': 'Fake NPU', 'NPU_DRIVER_VERSION': '3053'}[name]

        def import_model(self, stream, device):
            assert device == 'NPU' and stream.read() == b'npu blob'
            return Compiled(task.manifest)

    openvino = ModuleType('openvino')
    openvino.Core, openvino.AsyncInferQueue, openvino.get_version = Core, Queue, lambda: 'fake-2026.3.1'
    monkeypatch.setitem(sys.modules, 'openvino', openvino)
    monkeypatch.delitem(sys.modules, 'intel_execute_graph', raising=False)
    spec = importlib.util.spec_from_file_location('intel_npu_bench_worker', ROOT / 'scripts' / 'intel_npu_bench_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'execution-request.json'
    path.write_text(task.model_dump_json())
    return module, task, path


def test_worker_loads_saves_cases_and_runs_both_window_modes(worker_module, tmp_path):
    module, task, path = worker_module
    events = []
    worker = module.Worker(events.append)
    info = worker.load(str(path), task.artifacts['compiled.blob'])
    assert info == {'device': 'Fake NPU', 'architecture': '3720', 'execution_devices': ['NPU'], 'driver': '3053',
                    'openvino': 'fake-2026.3.1', 'optimal_nireq': 1}
    assert worker.cases(str(tmp_path / 'out')) == {'cases': 2}
    for case_id, encoded in task.cases.items():
        expected = sum(np.frombuffer(__import__('base64').b64decode(encoded[n].data_base64), np.float32) for n in ('x', 'y'))
        with np.load(tmp_path / 'out' / f'{case_id}.npz') as saved:
            assert np.array_equal(saved['output'], expected)
    assert worker.warmup(5) == {'warmup': 5} and worker.rounds(2, 10)['mean_s'] > 0
    sync = worker.window('sync', 0.01, 0.05, 0.01)
    assert [e['event'] for e in events] == ['begin', 'end'] and sync['event'] == 'done' and sync['calls'] >= 16
    assert sync['nireq'] == 1 and events[1]['calls'] == sync['calls']
    events.clear()
    asynchronous = worker.window('async', 0.01, 0.05, 0.01, nireq=3)
    queue = worker.queues[3]
    assert len(queue) == 3 and queue.started >= asynchronous['calls'] and queue.waits >= 2  # drained inside and after
    assert asynchronous['nireq'] == 3 and [e['event'] for e in events] == ['begin', 'end']


def test_serve_reports_errors_and_quits(worker_module):
    module, _, _ = worker_module
    replies = []
    lines = [json.dumps({'cmd': 'window', 'mode': 'sync', 'warm': 0, 'seconds': 0, 'tail': 0}),
             json.dumps({'cmd': 'shutdown'}), json.dumps({'cmd': 'quit'}), json.dumps({'cmd': 'warmup', 'count': 1})]
    module.serve(module.Worker(replies.append), lines, replies.append)
    assert 'AttributeError' in replies[0]['error']  # nothing loaded yet
    assert 'unknown command' in replies[1]['error'] and replies[2] == {'ok': True} and len(replies) == 3
