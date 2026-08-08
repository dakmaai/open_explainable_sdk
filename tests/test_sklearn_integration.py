from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression


def test_sklearn_pipeline_explain_with_dataframe(client):
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 0.0, 1.0, 0.0]})
    y = np.array([0, 0, 1, 1])

    @client.track_training
    def train(X_train, y_train):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        return model

    model = train(X, y)

    @client.explain(threshold=0.5)
    def score(model, row):
        return model.predict_proba(row)

    row = X.iloc[[2]]
    result = score(model, row)

    assert result.decision_output.label in {"APPROVED", "DECLINED"}
    assert 0.0 <= result.decision_output.score <= 1.0
    assert result.explain.audit_trail_id.startswith("ax-")
