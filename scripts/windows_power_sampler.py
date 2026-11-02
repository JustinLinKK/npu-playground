"""Sample SoC and GPU power for research/benchmark_kernel_energy.py (Windows-native Python, standard library only).

SoC: Windows Energy Meter (RAPL) counters through PDH, readable without administrator rights: package (PKG), CPU
cores (PP0), integrated graphics (PP1) and DRAM, cumulative in picowatt-hours, plus the meter's millisecond clock and
total CPU utility as a background-load signal. GPU: NVML instantaneous board power (field 186), the 1 s average,
P-state, SM clock and utilization. NVML's total-energy counter is about 25x wrong on the RTX 5070 used here, so it is
never read. Method: docs/time-energy-measurement.md.

Commands are JSON lines on stdin; each gets one JSON line back carrying the sampler clock 't' (perf_counter, s):
  {"cmd": "ping"}, {"cmd": "mark", "label": ...}, {"cmd": "latest"}, {"cmd": "flush", "path": <Windows path>},
  {"cmd": "stop"}.
latest returns the newest sample (the orchestrator waits on its GPU P-state); flush appends the samples collected
since the previous flush to a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
from pathlib import Path
import sys
import threading
import time

FIELDS = ['t', 'pkg_pwh', 'pp0_pwh', 'pp1_pwh', 'dram_pwh', 'meter_ms', 'cpu_util',
          'gpu_w', 'gpu_avg_w', 'gpu_pstate', 'gpu_sm_mhz', 'gpu_util']
NVML_FI_DEV_POWER_INSTANT = 186


class RawCounter(ctypes.Structure):  # PDH_RAW_COUNTER
    _fields_ = [('CStatus', ctypes.c_uint32), ('TimeLow', ctypes.c_uint32), ('TimeHigh', ctypes.c_uint32),
                ('FirstValue', ctypes.c_longlong), ('SecondValue', ctypes.c_longlong), ('MultiCount', ctypes.c_uint32)]


class FormattedCounter(ctypes.Structure):  # PDH_FMT_COUNTERVALUE with PDH_FMT_DOUBLE
    _fields_ = [('CStatus', ctypes.c_uint32), ('doubleValue', ctypes.c_double)]


class FieldValue(ctypes.Structure):  # nvmlFieldValue_t
    _fields_ = [('fieldId', ctypes.c_uint), ('scopeId', ctypes.c_uint), ('timestamp', ctypes.c_longlong),
                ('latencyUsec', ctypes.c_longlong), ('valueType', ctypes.c_int), ('nvmlReturn', ctypes.c_int),
                ('value', ctypes.c_ulonglong)]


class Utilization(ctypes.Structure):  # nvmlUtilization_t
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


def checked(status: int, what: str) -> None:
    if status:
        raise OSError(f'{what} failed with status 0x{status & 0xffffffff:08x}')


class EnergyMeter:
    """RAPL rails and CPU utility through pdh.dll."""
    RAILS = ('PKG', 'PP0', 'PP1', 'DRAM')

    def __init__(self):
        self.pdh = ctypes.WinDLL('pdh')
        self.query = ctypes.c_void_p()
        checked(self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)), 'PdhOpenQueryW')
        self.energy = {rail: self.add(f'\\Energy Meter(RAPL_Package0_{rail})\\Energy') for rail in self.RAILS}
        self.time = self.add('\\Energy Meter(RAPL_Package0_PKG)\\Time')
        self.utility = self.add('\\Processor Information(_Total)\\% Processor Utility')
        self.pdh.PdhCollectQueryData(self.query)

    def add(self, path: str):
        handle = ctypes.c_void_p()
        checked(self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)), path)
        return handle

    def raw(self, handle) -> int:
        value, kind = RawCounter(), ctypes.c_uint32()
        checked(self.pdh.PdhGetRawCounterValue(handle, ctypes.byref(kind), ctypes.byref(value)), 'PdhGetRawCounterValue')
        return value.FirstValue

    def read(self) -> dict:
        self.pdh.PdhCollectQueryData(self.query)
        utility = FormattedCounter()
        status = self.pdh.PdhGetFormattedCounterValue(self.utility, 0x200, None, ctypes.byref(utility))
        sample = {f'{rail.lower()}_pwh': self.raw(handle) for rail, handle in self.energy.items()}
        return {**sample, 'meter_ms': self.raw(self.time), 'cpu_util': utility.doubleValue if status == 0 else None}


class Nvml:
    """Board power and state of GPU 0 through nvml.dll; never the total-energy counter."""

    def __init__(self, path: str = r'C:\Windows\System32\nvml.dll'):
        self.lib = ctypes.CDLL(path)
        checked(self.lib.nvmlInit_v2(), 'nvmlInit_v2')
        self.handle = ctypes.c_void_p()
        checked(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)), 'nvmlDeviceGetHandleByIndex_v2')

    def read(self) -> dict:
        field = FieldValue(fieldId=NVML_FI_DEV_POWER_INSTANT)
        self.lib.nvmlDeviceGetFieldValues(self.handle, 1, ctypes.byref(field))
        instant = field.value & 0xffffffff if field.valueType in (1, 5) else field.value  # 32-bit union members
        power, state, clock, utilization = ctypes.c_uint(), ctypes.c_int(), ctypes.c_uint(), Utilization()
        average = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power)) == 0
        pstate = self.lib.nvmlDeviceGetPerformanceState(self.handle, ctypes.byref(state)) == 0
        sm = self.lib.nvmlDeviceGetClockInfo(self.handle, 1, ctypes.byref(clock)) == 0  # NVML_CLOCK_SM
        busy = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)) == 0
        return {'gpu_w': instant / 1e3 if field.nvmlReturn == 0 else None,
                'gpu_avg_w': power.value / 1e3 if average else None, 'gpu_pstate': state.value if pstate else None,
                'gpu_sm_mhz': clock.value if sm else None, 'gpu_util': utilization.gpu if busy else None}


class Sampler:
    """Polls readers every `interval` seconds on a background thread and answers commands on the caller's thread."""

    def __init__(self, readers, interval: float, clock=time.perf_counter):
        self.readers, self.interval, self.clock = readers, interval, clock
        self.rows, self.marks, self.lock, self.stopped = [], [], threading.Lock(), threading.Event()
        self.last: dict = {}

    def sample(self) -> None:
        row = {'t': self.clock()}
        for reader in self.readers:
            row.update(reader.read())
        with self.lock:
            self.rows.append(row)
            self.last = row

    def loop(self) -> None:
        due = self.clock()
        while not self.stopped.is_set():
            self.sample()
            due += self.interval
            delay = due - self.clock()
            if delay > 0:
                self.stopped.wait(delay)
            else:
                due = self.clock()  # fell behind; do not burst

    def handle(self, command: dict) -> dict:
        now = self.clock()  # a mark's time is when its line was read
        name = command.get('cmd')
        if name in ('ping', 'mark'):
            if name == 'mark':
                self.marks.append((now, command.get('label')))
            return {'t': now}
        if name == 'latest':
            with self.lock:
                return {'t': now, 'sample': dict(self.last)}
        if name == 'flush':
            with self.lock:
                rows, self.rows = self.rows, []
            path = Path(command['path'])
            header = not path.exists()
            with path.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
                if header:
                    writer.writeheader()
                writer.writerows(rows)
            return {'t': now, 'rows': len(rows)}
        if name == 'stop':
            self.stopped.set()
            return {'t': now}
        raise ValueError(f'unknown command: {name}')


def serve(sampler: Sampler, lines, write) -> None:
    for line in lines:
        if not line.strip():
            continue
        try:
            reply = sampler.handle(json.loads(line))
        except (ValueError, KeyError, OSError) as exc:
            reply = {'error': f'{type(exc).__name__}: {exc}'}
        write(reply)
        if sampler.stopped.is_set():
            return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--interval', type=float, default=0.02, help='seconds between samples')
    parser.add_argument('--no-gpu', action='store_true', help='do not read NVML')
    args = parser.parse_args(argv)
    readers = [EnergyMeter()]
    gpu = None
    if not args.no_gpu:
        try:
            gpu = Nvml()
            readers.append(gpu)
        except OSError as exc:
            print(f'NVML unavailable, sampling the SoC only: {exc}', file=sys.stderr)
    sampler = Sampler(readers, args.interval)
    sampler.sample()  # fail before acknowledging if a reader is broken
    thread = threading.Thread(target=sampler.loop, daemon=True)
    thread.start()
    out = sys.stdout.buffer  # bytes: Windows text streams would translate line endings

    def write(reply):
        out.write((json.dumps(reply) + '\n').encode())
        out.flush()

    write({'ready': True, 't': sampler.clock(), 'gpu': gpu is not None, 'interval': args.interval})
    serve(sampler, iter(sys.stdin.buffer.readline, b''), write)
    sampler.stopped.set()
    thread.join()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
