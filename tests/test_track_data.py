from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


def test_track_data_records_schema_and_lineage(client):
    @client.track_data
    def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ratio"] = df["a"] / df["b"]
        return df

    raw = pd.DataFrame({"a": [10, 20], "b": [2, 4]})
    out = prepare_features(raw)

    assert "ratio" in out.columns
    report = client.last_data_report()
    assert report is not None
    assert report["type"] == "data_tracking"
    assert report["function"] == "prepare_features"
    assert "a" in report["input_schema"]
    assert "ratio" in report["output_schema"]
    assert report["lineage_added"]["ratio"] == 'df["a"] / df["b"]'
