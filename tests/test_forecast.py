"""Forecast explanations: the decision is the predicted value, not approve/decline."""

from __future__ import annotations

import numpy as np
import pytest

import dakma
from dakma_sdk.models import TopFactor
from tests.helpers import requires_sklearn_ml


def _spy_factors(client):
    seen = {}

    def spy(model, row, top_k=3, *, class_index=None):
        seen["class_index"] = class_index
        seen["row"] = row
        return [TopFactor(name="lag_1", value=4.0, impact=-0.5, direction="down")], None

    client._top_factors = spy
    return seen


def test_forecast_uses_feature_metrics_and_selected_horizon(tmp_path) -> None:
    dm = dakma.init(project="demand-forecast")
    seen = _spy_factors(dm)

    @dm.explain_forecast(horizon=1, counterfactual=True)
    def forecast(model, feature_metrics):
        return np.array([1.5, 3.25, 2.0])

    feature_metrics = {"lag_1": 4.0, "temp": 18.0}
    event = forecast(object(), feature_metrics)

    assert event.decision.value == "FORECAST"
    assert event.decision.forecast == pytest.approx(3.25)
    assert event.decision.horizon == 1
    assert event.decision.score == pytest.approx(3.25)
    assert event.decision.threshold is None
    assert event.decision.to_text() == "forecast 3.25 at horizon 1"
    assert seen["class_index"] == 1
    assert seen["row"] == feature_metrics
    assert event.explanation.metadata["feature_metrics"] == {"lag_1": 4.0, "temp": 18.0}
    assert event.explanation.plain_language.startswith("Forecast 3.25 at horizon 1")
    assert "forecast" in event.explanation.plain_language
    assert "approve" not in event.explanation.counterfactual.lower()

    report = event.to_report_dict()["decision_output"]
    assert report["forecast"] == pytest.approx(3.25)
    assert report["horizon"] == 1
    assert report["threshold"] is None
    markdown = event.as_markdown_table()
    assert "| forecast | 3.25 |" in markdown
    assert "| horizon | 1 |" in markdown
    assert "| lag_1 | 4.0 |" in markdown
    html = event.write_report(tmp_path / "forecast.html", format="html").read_text()
    assert "Feature metrics" in html
    assert "lag_1" in html


def test_forecast_keyword_feature_metrics_and_default_horizon() -> None:
    dm = dakma.init(project="demand-forecast-default")
    seen = _spy_factors(dm)

    @dm.explain_forecast()
    def forecast(model, feature_metrics):
        return np.array([10.0, 20.0])

    event = forecast(object(), feature_metrics={"lag_1": 1.0})
    assert event.decision.forecast == pytest.approx(10.0)
    assert event.decision.horizon == 0
    assert seen["class_index"] == 0
    assert event.explanation.metadata["feature_metrics"]["lag_1"] == pytest.approx(1.0)


def test_forecast_horizon_outside_the_return_raises() -> None:
    dm = dakma.init(project="demand-forecast-horizon")

    @dm.explain_forecast(horizon=3)
    def forecast(model, feature_metrics):
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="horizon 3"):
        forecast(object(), {"lag_1": 1.0})


def test_explain_classification_is_unchanged() -> None:
    dm = dakma.init(project="still-classification")

    @dm.explain()
    def score(model, row):
        return np.array([[0.2, 0.8]])

    event = score(object(), {"lag_1": 4.0})
    assert event.decision.value == "APPROVED"
    assert event.decision.forecast is None
    assert event.decision.threshold == pytest.approx(0.5)
    assert "feature_metrics" not in event.explanation.metadata


@requires_sklearn_ml
def test_random_forest_regressor_explains_the_forecast() -> None:
    import shap
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    lag_1 = rng.normal(size=60)
    temp = rng.normal(size=60)
    target = 2.0 * lag_1 + 0.1 * temp
    import pandas as pd

    frame = pd.DataFrame({"lag_1": lag_1, "temp": temp})
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(frame, target)
    feature_metrics = frame.iloc[[0]]
    expected = float(model.predict(feature_metrics)[0])

    dm = dakma.init(project="rf-forecast")

    @dm.explain_forecast()
    def forecast(regressor, feature_metrics):
        return regressor.predict(feature_metrics)

    event = forecast(model, feature_metrics)
    assert event.decision.forecast == pytest.approx(expected)
    assert event.decision.horizon == 0
    assert event.explanation.status == "success"
    assert event.explanation.metadata["feature_metrics"]["lag_1"] == pytest.approx(float(feature_metrics.iloc[0]["lag_1"]))

    raw = np.asarray(shap.Explainer(model)(feature_metrics).values[0], dtype=float)
    top = feature_metrics.columns[int(np.argmax(np.abs(raw)))]
    assert event.explanation.factors[0].name == top
    assert event.explanation.factors[0].impact == pytest.approx(float(raw[list(feature_metrics.columns).index(top)]))
