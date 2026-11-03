"""
Explain a demand forecast from lag and calendar feature metrics.

The feature metrics for the forecast origin are passed into the regressor. The
decision is the forecast value, and SHAP attributes that value.

Run from the ``examples/`` directory:

    cd examples && python forecast_example.py

Requires optional dependencies: ``pip install "dakma-sdk[ml]"`` (pandas, scikit-learn, shap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import dakma


REPORT_DIR = Path(__file__).resolve().parent / "reports_forecast"
rng = np.random.default_rng(0)
n_rows = 120
lag_1 = rng.normal(loc=10.0, scale=2.0, size=n_rows)
lag_7 = rng.normal(loc=10.0, scale=2.0, size=n_rows)
temp = rng.normal(loc=18.0, scale=5.0, size=n_rows)
demand = 3.0 * lag_1 + 0.5 * lag_7 + 0.2 * temp + rng.normal(scale=0.5, size=n_rows)
history = pd.DataFrame({"lag_1": lag_1, "lag_7": lag_7, "temp": temp})

model = RandomForestRegressor(n_estimators=40, random_state=0).fit(history, demand)
feature_metrics = history.iloc[[-1]]

dm = dakma.init(project="demand-forecast", regulation="internal-policy", risk_level="medium")
dm.register_shap_feature_importance(model, history, max_samples=min(80, len(history)))


@dm.explain_forecast(counterfactual=True)
def forecast(regressor, feature_metrics):
    return regressor.predict(feature_metrics)


def main() -> None:
    result = forecast(model, feature_metrics)
    print(result.decision.to_text())
    print(result.explanation.plain_language)
    print(result.explanation.counterfactual)
    print("feature metrics passed to the model:")
    for name, value in result.explanation.metadata["feature_metrics"].items():
        print(f"  {name}: {value:.3f}")
    for factor in result.explanation.factors:
        print(f"  {factor.name:<8} impact={factor.impact:+.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = result.write_report(REPORT_DIR / "forecast.html", title="Demand forecast", format="html")
    print()
    print("--- Downloadable reports written ---")
    print(f"  {path}")


if __name__ == "__main__":
    main()
