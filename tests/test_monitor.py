from __future__ import annotations

import pytest


def test_monitor_computes_accuracy_and_drift(client):
    report = client.monitor(
        y_true=[1, 0, 1, 1, 0],
        y_pred=[1, 0, 0, 1, 0],
        expected_rate=0.6,
    )
    assert report["accuracy"] == pytest.approx(0.8)
    assert report["positive_prediction_rate"] == pytest.approx(0.4)
    assert report["drift_from_expected_rate"] == pytest.approx(0.2)
    assert report["group_positive_rates"] == {}


def test_monitor_group_rates(client):
    report = client.monitor(
        y_true=[1, 0, 1, 0],
        y_pred=[1, 1, 0, 0],
        protected_feature=["a", "a", "b", "b"],
    )
    assert report["group_positive_rates"]["a"] == pytest.approx(1.0)
    assert report["group_positive_rates"]["b"] == pytest.approx(0.0)
