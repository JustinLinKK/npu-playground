"""Pure helpers for the kernel time and energy benchmark (method: docs/time-energy-measurement.md).

Power comes from cumulative energy counters (Windows Energy Meter, RAPL) and from sampled GPU board power; both are
read on the sampler's clock. A run window's power is compared with the mean of its neighbouring idle windows, which
cancels linear drift, and paired differences get Student-t confidence intervals. Only numpy is required, so the
WSL orchestrator and the Windows NPU worker share this module.
"""
from __future__ import annotations

import math
import random

import numpy as np

PWH_TO_J = 3.6e-9  # Energy Meter counters are cumulative picowatt-hours
RAILS = ('pkg', 'pp0', 'pp1')  # package, CPU cores, integrated graphics; DRAM reads 0 on this platform
# NVML P-state of an idle GPU. After work the RTX 5070 holds P1 (~23 W board, and ~1.8 W less uncore) for 5-7 s and
# reaches P8 (~11 W) 7-9 s after the last launch, so an idle window is a baseline only once the GPU is back at P8.
GPU_IDLE_PSTATE = 8
SETTLED_FRACTION = 0.9
# Student-t quantiles by degrees of freedom: two-sided 95% (0.975), one-sided 95% and 80%. No scipy in the GPU venv.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       15: 1.753, 20: 1.725, 30: 1.697, 60: 1.671, 120: 1.658}
T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879,
       15: 0.866, 20: 0.860, 30: 0.854, 60: 0.848, 120: 0.845}


def quantile(table: dict[int, float], df: int) -> float:
    """Tabulated quantile at the largest listed df not above df (conservative between rows)."""
    listed = [key for key in table if key <= df]
    return table[max(listed)] if listed else math.inf


def counter_power(t, joules, start: float, end: float) -> float:
    """Mean power over [start, end] from a cumulative energy counter (J) sampled at times t (s)."""
    t, joules = np.asarray(t, float), np.asarray(joules, float)
    if not t[0] <= start < end <= t[-1]:
        raise ValueError('window lies outside the power trace')
    return float((np.interp(end, t, joules) - np.interp(start, t, joules)) / (end - start))


def sampled_power(t, watts, start: float, end: float) -> float:
    """Mean of a sampled power signal over [start, end]: trapezoid integral with interpolated window edges."""
    t, watts = np.asarray(t, float), np.asarray(watts, float)
    if not t[0] <= start < end <= t[-1]:
        raise ValueError('window lies outside the power trace')
    inside = (t > start) & (t < end)
    times = np.concatenate([[start], t[inside], [end]])
    values = np.concatenate([[np.interp(start, t, watts)], watts[inside], [np.interp(end, t, watts)]])
    return float(np.trapezoid(values, times) / (end - start))


def window_power(trace: dict, start: float, end: float) -> dict[str, float]:
    """Mean power of each rail over one window. trace holds arrays t (s), <rail>_j (cumulative J) and optionally
    gpu_w (sampled board power, W). uncore = pkg - pp0 - pp1 is the package outside the CPU cores and graphics, which
    contains the NPU."""
    power = {rail: counter_power(trace['t'], trace[rail + '_j'], start, end) for rail in RAILS}
    power['uncore'] = power['pkg'] - power['pp0'] - power['pp1']
    if trace.get('gpu_w') is not None:
        power['gpu'] = sampled_power(trace['t'], trace['gpu_w'], start, end)
    return power


def gpu_settled(trace: dict, start: float, end: float) -> bool:
    """True if the GPU sat in its idle P-state for at least SETTLED_FRACTION of [start, end] (or no GPU is traced)."""
    if trace.get('gpu_pstate') is None:
        return True
    t = np.asarray(trace['t'], float)
    states = np.asarray(trace['gpu_pstate'], float)[(t >= start) & (t <= end)]
    return bool(len(states)) and float(np.mean(states >= GPU_IDLE_PSTATE)) >= SETTLED_FRACTION


def paired_deltas(windows: list[dict]) -> list[dict]:
    """For each run window (session order), its power minus the mean of the nearest usable idle windows before and
    after. Windows are dicts with 'kind' ('idle' or 'run') and 'power'; a window with 'settled' False is skipped (an
    idle window is then not a baseline, a run is dropped). Other keys are copied to the pair."""
    def usable(w):
        return w['kind'] == 'idle' and w.get('settled', True)

    pairs = []
    for index, window in enumerate(windows):
        if window['kind'] != 'run' or not window.get('settled', True):
            continue
        before = next((w for w in reversed(windows[:index]) if usable(w)), None)
        after = next((w for w in windows[index + 1:] if usable(w)), None)
        if before is None or after is None:
            continue
        idle = {rail: (before['power'][rail] + after['power'][rail]) / 2 for rail in window['power']}
        pairs.append({**{k: v for k, v in window.items() if k != 'power'}, 'power': window['power'], 'idle': idle,
                      'delta': {rail: window['power'][rail] - idle[rail] for rail in window['power']}})
    return pairs


def estimate(values) -> dict:
    """Mean of paired differences with standard error, 95% CI, significance (CI excludes 0), the effect detectable
    with 80% power (MDE80) and a one-sided 95% upper bound."""
    x = np.asarray(values, float)
    count, mean = len(x), float(x.mean())
    if count < 2:
        return {'n': count, 'mean': mean, 'sd': None, 'se': None, 'ci95': None, 'significant': False,
                'mde80': None, 'upper95': None}
    sd = float(x.std(ddof=1))
    se, df = sd / math.sqrt(count), count - 1
    half = quantile(T975, df) * se
    return {'n': count, 'mean': mean, 'sd': sd, 'se': se, 'ci95': [mean - half, mean + half],
            'significant': bool(abs(mean) > half), 'mde80': (quantile(T975, df) + quantile(T80, df)) * se,
            'upper95': mean + quantile(T95, df) * se}


def schedule(modes: list[str], cycles: int, seed: int) -> list[str]:
    """Run order for one task: every mode once per cycle, shuffled per cycle with a fixed seed."""
    rng, order = random.Random(seed), []
    for _ in range(cycles):
        cycle = list(modes)
        rng.shuffle(cycle)
        order += cycle
    return order


def fixed_window(step, seconds: float, clock, check_every: int = 16, finish=None) -> dict:
    """Call step() back to back for `seconds`, reading the clock every `check_every` calls. finish() (a synchronize
    or wait_all) completes queued work before the end time is read, so seconds/calls is the time of one run."""
    start = clock()
    deadline, middle, calls, half = start + seconds, start + seconds / 2, 0, None
    while True:
        for _ in range(check_every):
            step()
        calls += check_every
        now = clock()
        if half is None and now >= middle:
            half = (calls, now - start)
        if now >= deadline:
            break
    if finish is not None:
        finish()
    elapsed = clock() - start
    return {'calls': calls, 'seconds': elapsed, 'per_call_s': elapsed / calls, 'half': half}


def half_rates(window: dict) -> tuple[float, float]:
    """Calls per second in the first and second half of a fixed window; they differ if the device changes power state."""
    calls, seconds = window['half']
    return calls / seconds, (window['calls'] - calls) / (window['seconds'] - seconds)


def run_window(step, warm: float, seconds: float, tail: float, clock, begin=None, end=None, finish=None) -> dict:
    """Keep a device busy around a counted fixed window: `warm` seconds before it, then begin(), the counted window,
    end(result), and `tail` seconds after it. The power window marked by begin and end therefore lies inside the busy
    span even when markers arrive a few milliseconds late."""
    stop = clock() + warm
    while clock() < stop:
        step()
    if begin is not None:
        begin()
    result = fixed_window(step, seconds, clock, finish=finish)
    if end is not None:
        end(result)
    stop = clock() + tail
    while clock() < stop:
        step()
    if finish is not None:
        finish()
    return result


def rounds(step, count: int, calls: int, clock) -> dict:
    """Per-call times (s) over `count` rounds of `calls` timed calls: mean, percentiles and the spread of round means."""
    samples, means = [], []
    for _ in range(count):
        times = []
        for _ in range(calls):
            started = clock()
            step()
            times.append(clock() - started)
        samples += times
        means.append(sum(times) / calls)
    array = np.asarray(samples)
    return {'mean_s': float(array.mean()), 'p50_s': float(np.percentile(array, 50)),
            'p95_s': float(np.percentile(array, 95)), 'p99_s': float(np.percentile(array, 99)),
            'round_means_s': means, 'round_spread': (max(means) - min(means)) / float(array.mean())}
