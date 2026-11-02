import csv
import ctypes
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('windows_power_sampler', ROOT / 'scripts' / 'windows_power_sampler.py')
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)


class Rails:
    def __init__(self):
        self.energy = 0

    def read(self):
        self.energy += 1000
        return {'pkg_pwh': self.energy, 'pp0_pwh': self.energy // 2, 'pp1_pwh': 0, 'dram_pwh': 0, 'meter_ms': 5, 'cpu_util': 3.0}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.02
        return self.now


def test_commands_mark_on_the_sampler_clock_and_flush_appends_csv(tmp_path):
    sampler = sampler_module.Sampler([Rails()], 0.02, clock=Clock())
    sampler.sample()
    sampler.sample()
    mark = sampler.handle({'cmd': 'mark', 'label': 'run'})
    assert mark['t'] > 0 and sampler.marks == [(mark['t'], 'run')]
    path = tmp_path / 'trace.csv'
    assert sampler.handle({'cmd': 'flush', 'path': str(path)})['rows'] == 2
    sampler.sample()
    assert sampler.handle({'cmd': 'flush', 'path': str(path)})['rows'] == 1
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['pkg_pwh']) for row in rows] == [1000, 2000, 3000] and rows[0]['gpu_w'] == ''
    assert list(rows[0]) == sampler_module.FIELDS


def test_latest_returns_the_newest_sample_even_after_a_flush(tmp_path):
    sampler = sampler_module.Sampler([Rails()], 0.02, clock=Clock())
    assert sampler.handle({'cmd': 'latest'})['sample'] == {}
    sampler.sample()
    sampler.sample()
    sampler.handle({'cmd': 'flush', 'path': str(tmp_path / 'trace.csv')})
    assert sampler.handle({'cmd': 'latest'})['sample']['pkg_pwh'] == 2000


def test_serve_reports_bad_commands_and_stops(tmp_path):
    sampler, replies = sampler_module.Sampler([Rails()], 0.02, clock=Clock()), []
    lines = [json.dumps({'cmd': 'ping'}), json.dumps({'cmd': 'bogus'}), '', json.dumps({'cmd': 'stop'}),
             json.dumps({'cmd': 'ping'})]
    sampler_module.serve(sampler, lines, replies.append)
    assert 't' in replies[0] and 'unknown command' in replies[1]['error'] and len(replies) == 3
    assert sampler.stopped.is_set()


class FakeNvml:
    """Writes values through the ctypes references the sampler passes; any unexpected NVML call fails."""

    def nvmlDeviceGetFieldValues(self, handle, count, field):
        value = field._obj
        assert value.fieldId == sampler_module.NVML_FI_DEV_POWER_INSTANT
        value.value, value.valueType, value.nvmlReturn = (0xdead << 32) | 41250, 1, 0  # 32-bit union member
        return 0

    def nvmlDeviceGetPowerUsage(self, handle, power):
        power._obj.value = 40000
        return 0

    def nvmlDeviceGetPerformanceState(self, handle, state):
        state._obj.value = 2
        return 0

    def nvmlDeviceGetClockInfo(self, handle, kind, clock):
        assert kind == 1
        clock._obj.value = 2500
        return 0

    def nvmlDeviceGetUtilizationRates(self, handle, utilization):
        utilization._obj.gpu = 37
        return 0

    def __getattr__(self, name):
        raise AssertionError(f'unexpected NVML call {name}')


def test_nvml_reader_converts_instant_power_and_masks_the_union():
    reader = sampler_module.Nvml.__new__(sampler_module.Nvml)
    reader.lib, reader.handle = FakeNvml(), ctypes.c_void_p()
    assert reader.read() == {'gpu_w': 41.25, 'gpu_avg_w': 40.0, 'gpu_pstate': 2, 'gpu_sm_mhz': 2500, 'gpu_util': 37}


class FakePdh:
    def PdhCollectQueryData(self, query):
        return 0

    def PdhGetFormattedCounterValue(self, handle, kind, _, value):
        value._obj.doubleValue = 4.5
        return 0

    def PdhGetRawCounterValue(self, handle, kind, value):
        value._obj.FirstValue = handle.value * 100
        return 0


def test_energy_meter_reads_raw_counters_per_rail():
    reader = sampler_module.EnergyMeter.__new__(sampler_module.EnergyMeter)
    reader.pdh, reader.query = FakePdh(), ctypes.c_void_p()
    reader.energy = {rail: ctypes.c_void_p(index + 1) for index, rail in enumerate(reader.RAILS)}
    reader.time, reader.utility = ctypes.c_void_p(9), ctypes.c_void_p(10)
    assert reader.read() == {'pkg_pwh': 100, 'pp0_pwh': 200, 'pp1_pwh': 300, 'dram_pwh': 400, 'meter_ms': 900, 'cpu_util': 4.5}


def test_structures_match_the_windows_layouts():
    assert ctypes.sizeof(sampler_module.RawCounter) == 40 and sampler_module.RawCounter.FirstValue.offset == 16
    assert sampler_module.FormattedCounter.doubleValue.offset == 8
    assert sampler_module.FieldValue.value.offset == 32


def test_checked_raises_with_the_status():
    with pytest.raises(OSError, match='0xc0000bb8'):
        sampler_module.checked(0xC0000BB8, 'PdhAddEnglishCounterW')
