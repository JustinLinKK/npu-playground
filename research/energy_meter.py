"""Pure helpers for the kernel time and energy benchmark (method: docs/time-energy-measurement.md).

Power comes from cumulative energy counters (Windows Energy Meter, RAPL) and from sampled GPU board power; both are
read on the sampler's clock. A run window's power is compared with the idle power interpolated in time between its
neighbouring idle windows, which cancels linear drift, and paired differences get Student-t confidence intervals. Only numpy is required, so the
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


def idle_pstate_fraction(trace: dict, start: float, end: float) -> float | None:
    """Fraction of the samples in [start, end] with the GPU in its idle P-state; None if no GPU is traced."""
    if trace.get('gpu_pstate') is None:
        return None
    t = np.asarray(trace['t'], float)
    states = np.asarray(trace['gpu_pstate'], float)[(t >= start) & (t <= end)]
    return float(np.mean(states >= GPU_IDLE_PSTATE)) if len(states) else 0.0


def gpu_settled(trace: dict, start: float, end: float) -> bool:
    """True if the GPU sat in its idle P-state for at least SETTLED_FRACTION of [start, end] (or no GPU is traced)."""
    fraction = idle_pstate_fraction(trace, start, end)
    return fraction is None or fraction >= SETTLED_FRACTION


def usable_idle(window: dict) -> bool:
    return window['kind'] == 'idle' and window.get('settled', True)


def midpoint(window: dict) -> float | None:
    return (window['start'] + window['end']) / 2 if 'start' in window and 'end' in window else None


def neighbours(windows: list[dict], index: int) -> tuple[dict | None, dict | None]:
    """The nearest usable (settled) idle windows before and after windows[index]."""
    before = next((w for w in reversed(windows[:index]) if usable_idle(w)), None)
    after = next((w for w in windows[index + 1:] if usable_idle(w)), None)
    return before, after


def baseline(before: dict, after: dict, at: float | None) -> dict[str, float]:
    """Idle power per rail at time `at`, interpolated linearly between two idle windows' midpoints (midway if any
    time is unknown)."""
    times = midpoint(before), at, midpoint(after)
    weight = 0.5 if None in times else (times[1] - times[0]) / (times[2] - times[0])
    return {rail: before['power'][rail] + weight * (after['power'][rail] - before['power'][rail])
            for rail in before['power']}


def paired_deltas(windows: list[dict]) -> list[dict]:
    """For each run window (session order), its power minus the idle power interpolated linearly in time, at the run's
    midpoint, between the nearest usable idle windows before and after; this cancels linear drift however unevenly the
    windows are spaced (the plain mean of the two does so only for a run exactly midway). Windows are dicts with 'kind'
    ('idle' or 'run'), 'power' and, for the interpolation, 'start' and 'end' (without them the run counts as midway).
    A window with 'settled' False is skipped (an idle window is then not a baseline, a run is dropped). Other keys are
    copied to the pair."""
    pairs = []
    for index, window in enumerate(windows):
        if window['kind'] != 'run' or not window.get('settled', True):
            continue
        before, after = neighbours(windows, index)
        if before is None or after is None:
            continue
        idle = baseline(before, after, midpoint(window))
        times = midpoint(before), midpoint(window), midpoint(after)
        pairs.append({**{k: v for k, v in window.items() if k != 'power'}, 'power': window['power'], 'idle': idle,
                      'baseline_weight': 0.5 if None in times else (times[1] - times[0]) / (times[2] - times[0]),
                      'delta': {rail: window['power'][rail] - idle[rail] for rail in window['power']}})
    return pairs


def burst_energies(trace: dict, windows: list[dict]) -> list[dict]:
    """For each 'bursts' window (an intermittent load whose 'marks' split it into whole periods), the energy per
    period above the idle power interpolated to each period's midpoint, in J per rail, averaged over its periods.
    A period holds one burst and everything after it until the next burst, so a device's post-burst power tail is
    charged to the burst that caused it."""
    found = []
    for index, window in enumerate(windows):
        if window['kind'] != 'bursts' or not window.get('settled', True):
            continue
        before, after = neighbours(windows, index)
        if before is None or after is None:
            continue
        periods = []
        for a, b in zip(window['marks'], window['marks'][1:]):
            power, idle = window_power(trace, a, b), baseline(before, after, (a + b) / 2)
            periods.append({rail: (power[rail] - idle[rail]) * (b - a) for rail in power})
        found.append({**{k: v for k, v in window.items() if k != 'marks'},
                      'idle': baseline(before, after, midpoint(window)),
                      'energy_j': {rail: float(np.mean([p[rail] for p in periods])) for rail in periods[0]}})
    return found


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


def ratio(numerator: dict, denominator: dict) -> dict | None:
    """Ratio of two independent estimates (energy per inference of two devices on one task) with a 95% CI from the
    delta method on the log scale, df = the smaller n - 1. None unless both are significant and positive."""
    if not all(e and e['significant'] and e['mean'] > 0 for e in (numerator, denominator)):
        return None
    value = numerator['mean'] / denominator['mean']
    se = math.hypot(numerator['se'] / numerator['mean'], denominator['se'] / denominator['mean'])
    half = quantile(T975, min(numerator['n'], denominator['n']) - 1) * se
    return {'ratio': value, 'ci95': [value * math.exp(-half), value * math.exp(half)], 'se_log': se}


def geometric_mean(values) -> dict:
    """Geometric mean of positive ratios across tasks, with the range and a 95% t-interval on the log scale (the
    spread between tasks). Ratios of arithmetic means would weight the slowest tasks most."""
    logs = np.log(np.asarray(values, float))
    out = {'n': len(logs), 'geomean': float(np.exp(logs.mean())), 'min': float(np.exp(logs.min())),
           'max': float(np.exp(logs.max())), 'ci95': None}
    if len(logs) >= 2:
        half = quantile(T975, len(logs) - 1) * float(logs.std(ddof=1)) / math.sqrt(len(logs))
        out['ci95'] = [float(np.exp(logs.mean() - half)), float(np.exp(logs.mean() + half))]
    return out


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
    if finish is not None:
        finish()  # drain queued warm-up work, which would otherwise run uncounted inside the window
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


def bursts(step, calls: int, period: float, count: int, clock, sleep, finish=None, begin=None, end=None) -> dict:
    """Intermittent load: at the start of each of `count` periods, begin() and a burst of `calls` step() calls that
    finish() completes; the device then idles until the next period. end() marks the end of the last period, so the
    marks split the load into whole periods, each holding one burst and the idle time (and any power tail) after it."""
    start, durations = clock(), []

    def wait_until(target):
        while (now := clock()) < target:
            sleep(min(target - now, 0.05))

    for index in range(count):
        wait_until(start + index * period)
        if begin is not None:
            begin()
        started = clock()
        for _ in range(calls):
            step()
        if finish is not None:
            finish()
        durations.append(clock() - started)
    wait_until(start + count * period)
    if end is not None:
        end()
    return {'calls': calls, 'period': period, 'count': count, 'burst_s': durations,
            'overrun': bool(max(durations) > period)}


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
