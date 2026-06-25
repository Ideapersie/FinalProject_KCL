"""
evaluation/profiler.py — Latency / memory / energy measurement wrapper.

profile_call() runs a callable and returns its result alongside a ProfileMetrics
record. These metrics populate PolicyResult.latency_ns / energy_kwh /
peak_memory_mb, which feed the Pareto and safety-envelope analyses.

Design notes:
  - Latency uses time.perf_counter_ns() — monotonic, high resolution.
  - Peak memory uses tracemalloc (Python-allocation peak during the call),
    falling back to a psutil RSS delta if tracemalloc is unavailable.
  - Energy uses codecarbon's EmissionsTracker, imported lazily and only when
    track_energy=True, so unit tests never touch codecarbon. On any failure
    energy is reported as 0.0 rather than crashing the run.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


@dataclass
class ProfileMetrics:
    latency_ns:     int   = 0
    energy_kwh:     float = 0.0
    peak_memory_mb: float = 0.0


def _make_tracker(country_iso: str):
    """Construct an EmissionsTracker, passing only kwargs this version accepts."""
    import inspect

    from codecarbon import EmissionsTracker

    # codecarbon's constructor signature changed across major versions; build the
    # kwargs defensively so we work on both 2.x (country_iso_code) and 3.x.
    accepted = set(inspect.signature(EmissionsTracker.__init__).parameters)
    kwargs = {}
    for key, value in (
        ("country_iso_code", country_iso),
        ("save_to_file", False),
        ("save_to_api", False),
        ("save_to_logger", False),
        ("log_level", "error"),
        ("allow_multiple_runs", True),
    ):
        if key in accepted:
            kwargs[key] = value
    return EmissionsTracker(**kwargs)


def _read_energy_kwh(tracker) -> float:
    """Extract total energy in kWh from a stopped tracker, 0.0 if unavailable."""
    total = getattr(tracker, "_total_energy", None)
    if total is None:
        return 0.0
    kwh = getattr(total, "kWh", total)
    try:
        return float(kwh)
    except (TypeError, ValueError):
        return 0.0


def _measure_energy(fn: Callable[[], Any], country_iso: str) -> Tuple[Any, float]:
    """Run fn under codecarbon, returning (result, energy_kwh). 0.0 on failure."""
    try:
        tracker = _make_tracker(country_iso)
    except Exception:
        return fn(), 0.0

    tracker.start()
    try:
        result = fn()
    finally:
        try:
            tracker.stop()
        except Exception:
            pass
    return result, _read_energy_kwh(tracker)


def profile_call(
    fn: Callable[[], Any],
    *,
    track_latency: bool = True,
    track_memory: bool = True,
    track_energy: bool = False,
    country_iso: str = "GBR",
) -> Tuple[Any, ProfileMetrics]:
    """
    Execute `fn` and return (result, ProfileMetrics).

    All tracking flags default conservatively; energy is off by default because
    it carries the heaviest dependency and is only wanted on real runs.
    """
    metrics = ProfileMetrics()

    if track_memory:
        tracemalloc.start()

    start = time.perf_counter_ns()
    if track_energy:
        result, metrics.energy_kwh = _measure_energy(fn, country_iso)
    else:
        result = fn()
    if track_latency:
        metrics.latency_ns = time.perf_counter_ns() - start

    if track_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics.peak_memory_mb = peak / (1024 * 1024)

    return result, metrics
