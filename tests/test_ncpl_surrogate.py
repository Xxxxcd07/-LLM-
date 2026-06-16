import numpy as np

from scripts import run_ncpl_surrogate


def test_build_data_by_size_loads_each_requested_size(monkeypatch):
    calls = []

    def fake_load(size):
        calls.append(size)
        return {"cosine_24000.csv": {"loss": np.array([1.0])}}

    monkeypatch.setattr(run_ncpl_surrogate, "load_size_data", fake_load)

    result = run_ncpl_surrogate.build_data_by_size(["25", "100"])

    assert calls == ["25", "100"]
    assert sorted(result) == ["100", "25"]


def test_evaluate_surrogate_writes_metric_stage_and_prediction_rows(monkeypatch):
    data_by_size = {
        "25": {
            "wsd_20000_24000.csv": {
                "lrs": np.array([0.2, 0.1]),
                "step": np.array([0, 1]),
                "loss": np.array([2.0, 1.0]),
            }
        }
    }

    class ConstantSurrogate:
        def predict(self, features):
            return np.array([2.0, 1.0])

    predictions = []
    monkeypatch.setattr(
        run_ncpl_surrogate,
        "write_prediction_csv",
        lambda *args: predictions.append(args),
    )

    metric_rows, stage_rows = run_ncpl_surrogate.evaluate_surrogate(
        surrogate=ConstantSurrogate(),
        data_by_size=data_by_size,
        eval_curves=["wsd_20000_24000.csv"],
    )

    assert metric_rows[0]["method"] == "ncpl_surrogate"
    assert metric_rows[0]["correction"] == "none"
    assert metric_rows[0]["train_curves"] == "cosine_24000.csv"
    assert stage_rows
    assert predictions[0][0] == run_ncpl_surrogate.RUN_ID
    assert predictions[0][2] == "ncpl_surrogate"
