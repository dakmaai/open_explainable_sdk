from __future__ import annotations

from tests.conftest import DummyModel


def test_track_training_records_model_metadata(client, dummy_model):
    @client.track_training
    def train(X, y):
        return dummy_model

    model = train([[1.0]], [1])
    assert model is dummy_model

    report = client.last_training_report()
    assert report is not None
    assert report["type"] == "training_tracking"
    assert report["function"] == "train"
    assert report["model_class"] == "DummyModel"
    assert report["model_params"]["n_estimators"] == 10
    assert report["metrics"]["training_score"] == 0.91
    assert report["compute_usage"] is not None
    assert "wall_time_ms" in report["compute_usage"]
