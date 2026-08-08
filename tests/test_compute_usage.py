from __future__ import annotations

from darsha_sdk.compute_usage import ComputeUsageRecorder, ComputeUsageSnapshot


def test_compute_usage_recorder_records_wall_time():
    with ComputeUsageRecorder() as rec:
        total = sum(i for i in range(1000))
    assert total == 499500
    assert rec.usage is not None
    assert rec.usage["wall_time_ms"] >= 0.0


def test_compute_usage_snapshot_as_dict():
    snap = ComputeUsageSnapshot(wall_time_ms=12.5, process_cpu_time_ms=8.0)
    d = snap.as_dict()
    assert d["wall_time_ms"] == 12.5
    assert d["process_cpu_time_ms"] == 8.0
