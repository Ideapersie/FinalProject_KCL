"""Unit tests for evaluation/profiler.py (no codecarbon import path)."""

import sys

from medrag_adaptive.evaluation.profiler import ProfileMetrics, profile_call


def test_returns_result_and_metrics():
    result, metrics = profile_call(lambda: 21 * 2)
    assert result == 42
    assert isinstance(metrics, ProfileMetrics)


def test_latency_positive():
    _, metrics = profile_call(lambda: sum(range(10000)))
    assert metrics.latency_ns > 0


def test_energy_off_by_default_is_zero():
    _, metrics = profile_call(lambda: 1)
    assert metrics.energy_kwh == 0.0


def test_energy_disabled_does_not_import_codecarbon(monkeypatch):
    # Ensure codecarbon is never imported when track_energy=False.
    monkeypatch.delitem(sys.modules, "codecarbon", raising=False)
    profile_call(lambda: 1, track_energy=False)
    assert "codecarbon" not in sys.modules


def test_memory_tracked_when_enabled():
    def allocate():
        return [0] * 100000
    _, metrics = profile_call(allocate, track_memory=True)
    assert metrics.peak_memory_mb >= 0.0


def test_latency_off_stays_zero():
    _, metrics = profile_call(lambda: 1, track_latency=False)
    assert metrics.latency_ns == 0
