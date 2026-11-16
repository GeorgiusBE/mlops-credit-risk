"""Exercise temporal isolation, trial accounting, model saving, and safe resume."""

import copy
import json
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import pytest

from feature_selection import train_baseline as baseline
from hyperparameter_tuning import tune


@pytest.fixture
def tuning_config():
    return tune.validate_config(baseline.load_config(tune.DEFAULT_CONFIG))


@pytest.fixture
def small_study(tmp_path, tuning_config):
    """Test labels are intentionally invalid and the test category is unique."""
    config = copy.deepcopy(tuning_config)
    config["model"].update(n_estimators=12, num_leaves=7, min_child_samples=2)
    config["early_stopping_rounds"] = 3
    config["split"] = {"train_start": 0, "train_end": 6, "validation_end": 8, "test_start": 9, "test_end": 10}
    config["windows"] = [
        {"name": "early", "train_end": 2, "validation_end": 4},
        {"name": "middle", "train_end": 4, "validation_end": 6},
        {"name": "late", "train_end": 6, "validation_end": 8},
    ]
    config["search_space"] = {"learning_rate": {"type": "categorical", "choices": [0.05, 0.1]}}
    config["feature_selection"].update(excluded_features=["deposit"], expected_feature_count=2)
    rng = np.random.default_rng(42)
    rows = []
    for week in range(11):
        for case in range(48):
            label = case % 2
            rows.append({"WEEK_NUM": week, "target": label if week < 9 else 999,
                         "amount": label + float(rng.normal(0, 0.7)), "deposit": float(case),
                         "industry": "test_only" if week >= 9 else ("later_development" if week >= 5 else "A")})
    pd.DataFrame(rows).to_parquet(tmp_path / "train_base_engineered.parquet", index=False)
    pd.DataFrame({"column_name": ["amount", "industry", "deposit"], "role": ["eligible"] * 3}).to_csv(
        tmp_path / "model_feature_eligibility.csv", index=False,
    )
    return config, tmp_path


def fingerprints(directory):
    return {"dataset_sha256": baseline.file_sha256(directory / "train_base_engineered.parquet"),
            "feature_inventory_sha256": baseline.file_sha256(directory / "model_feature_eligibility.csv")}


def read_artifact(client, run_id, name):
    return json.loads(Path(client.download_artifacts(run_id, name)).read_text())


def test_approved_feature_set_and_validation_boundaries(tuning_config):
    all_features = [name for columns in baseline.load_config(baseline.DEFAULT_CONFIG)["feature_groups"].values() for name in columns]
    excluded = tuning_config["feature_selection"]["excluded_features"]
    assert excluded == ["deposit_opening_earliest_days_ago", "deposit_opening_latest_days_ago", "deposit_opening_span_days"]
    assert len(set(all_features) - set(excluded)) == 17
    assert tuning_config["split"]["test_start"] == 78
    assert tuning_config["split"]["test_end"] == 91
    assert [(window["train_end"], window["validation_end"]) for window in tuning_config["windows"]] == [(35, 49), (49, 63), (63, 77)]


@pytest.mark.parametrize("mutation", ["test_overlap", "window_overlap", "duplicate_window", "unsafe_parameter", "invalid_control"])
def test_reject_invalid_plans(tuning_config, mutation):
    if mutation == "test_overlap":
        tuning_config["windows"][-1]["validation_end"] = 78
    elif mutation == "window_overlap":
        tuning_config["windows"][1]["train_end"] = 40
    elif mutation == "duplicate_window":
        tuning_config["windows"][1]["name"] = tuning_config["windows"][0]["name"]
    elif mutation == "unsafe_parameter":
        tuning_config["search_space"]["random_state"] = {"type": "int", "low": 1, "high": 42}
    else:
        tuning_config["search_space"]["learning_rate"]["low"] = 0.1
    with pytest.raises(ValueError):
        tune.validate_config(tuning_config)


def test_selected_projection_and_training_only_categories(small_study):
    config, directory = small_study
    features = tune.selected_features(config, directory / "model_feature_eligibility.csv")
    assert features == ["amount", "industry"]
    data = baseline.load_training_data(directory, tune.current_window_config(config, config["windows"][1]), 1, features=features)
    assert "deposit" not in data.train and "deposit" not in data.validation
    assert data.train["industry"].cat.categories.tolist() == ["A"]
    assert data.validation["industry"].isna().all()
    assert not data.train["target"].eq(999).any()
    assert not data.validation["target"].eq(999).any()
    frame, categorical = tune.load_refit_frame(directory, config, features, 1, fingerprints(directory))
    assert sorted(frame["WEEK_NUM"].unique()) == list(range(9))
    assert categorical == ["industry"]
    assert frame["industry"].cat.categories.tolist() == ["A", "later_development"]
    assert not frame["target"].eq(999).any()
    with pytest.raises(ValueError, match="subset"):
        baseline.load_training_data(directory, config, 1, features=["target"])
    config["feature_selection"]["expected_feature_count"] = 3
    with pytest.raises(ValueError, match="Feature count"):
        tune.selected_features(config, directory / "model_feature_eligibility.csv")


def test_equal_window_weighting_and_refit_rounds():
    summary = tune.summarize_results([
        {"metrics": {"gini_stability": 0.2, "roc_auc": 0.6}},
        {"metrics": {"gini_stability": 0.4, "roc_auc": 0.8}},
        {"metrics": {"gini_stability": 0.3, "roc_auc": 0.7}},
    ])
    assert summary[tune.PRIMARY_METRIC] == pytest.approx(0.3)
    assert summary["worst_validation_gini_stability"] == 0.2
    assert tune.refit_rounds([160, 210, 190]) == 190
    assert tune.refit_rounds([10, 11]) == 11
    for invalid in ([], [0, 1], [1.5, 2]):
        with pytest.raises(ValueError):
            tune.refit_rounds(invalid)


def test_dry_run_has_no_tracking_or_storage_side_effects(small_study, monkeypatch, capsys):
    config, directory = small_study
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", ["tune.py", "--dry-run", "--config", str(path),
                                    "--processed-dir", str(directory), "--storage-path", str(directory / "no.db")])
    def unexpected(*args, **kwargs):
        pytest.fail("A dry run must not contact tracking, train, or create a study")
    monkeypatch.setattr(tune.mlflow, "set_tracking_uri", unexpected)
    monkeypatch.setattr(tune.optuna, "create_study", unexpected)
    monkeypatch.setattr(tune.baseline, "load_training_data", unexpected)
    tune.main()
    output = json.loads(capsys.readouterr().out)
    assert output["features"] == ["amount", "industry"]
    assert output["refit"]["early_stopping"] is False
    assert not (directory / "no.db").exists()


def test_end_to_end_saves_only_final_model_and_resumes(small_study, monkeypatch):
    config, directory = small_study
    tracking_uri = f"sqlite:///{directory / 'tracking.db'}"
    # Keep even the model files outside the user's real MLflow artifact directories.
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment_id = client.create_experiment("test-tuning", artifact_location=(directory / "artifacts").as_uri())
    calls = []
    original_fit = tune.lgb.LGBMClassifier.fit
    def tracked_fit(self, X, y, *args, **kwargs):
        assert set(y.unique()) == {0, 1}
        assert list(X.columns) == ["amount", "industry"]
        calls.append(kwargs)
        return original_fit(self, X, y, *args, **kwargs)
    monkeypatch.setattr(tune.lgb.LGBMClassifier, "fit", tracked_fit)
    args = (config, directory, 1, tracking_uri, "test-tuning", "synthetic-study", directory / "optuna.db", 2)
    try:
        result = tune.run_tuning(*args)
        assert len(calls) == 7  # Two trials × three windows, then one refit.
        assert all("callbacks" in call and "eval_X" in call for call in calls[:6])
        assert calls[-1] == {}  # Final fit has neither early stopping nor a validation set.
        trials = client.search_runs([experiment_id], "tags.stage = 'hyperparameter_tuning_trial'")
        assert len(trials) == 2
        assert all(run.info.status == "FINISHED" and run.data.tags["optuna_trial_state"] == "COMPLETE" for run in trials)
        assert all(run.data.tags["mlflow.parentRunId"] == result["parent_run_id"] for run in trials)
        assert len(client.search_logged_models(experiment_ids=[experiment_id])) == 1
        assert all("mean_validation_gini_stability" in run.data.metrics for run in trials)
        parent = client.get_run(result["parent_run_id"])
        assert parent.data.tags["result_status"] == "refitted_test_not_evaluated"
        assert parent.data.metrics["trials_complete"] == 2
        final = client.get_run(result["run_id"])
        assert final.data.params["early_stopping"] == "False"
        assert not any("validation" in name for name in final.data.metrics)
        selected = read_artifact(client, result["parent_run_id"], "selected_trial.json")
        assert result["n_estimators"] == tune.refit_rounds(selected["best_iterations"])
        categories = read_artifact(client, result["run_id"], "categorical_levels.json")
        assert categories["industry"] == ["A", "later_development"]
        model = mlflow.lightgbm.load_model(result["model_uri"])
        frame, _ = tune.load_refit_frame(directory, config, ["amount", "industry"], 1, fingerprints(directory))
        predictions = model.predict(frame[["amount", "industry"]])
        assert len(predictions) == len(frame)
        assert np.isfinite(predictions).all()
        assert ((predictions >= 0) & (predictions <= 1)).all()
        resumed = tune.run_tuning(*args, resume=True)
        assert resumed == result
        assert len(calls) == 7 and len(client.search_logged_models(experiment_ids=[experiment_id])) == 1
        extended = tune.run_tuning(*args[:-1], 3, resume=True)
        assert extended["parent_run_id"] == result["parent_run_id"]
        assert len(client.search_runs([experiment_id], "tags.stage = 'hyperparameter_tuning_trial'")) == 3
        assert len(calls) in (10, 11)  # A new winner triggers one more refit.
        changed = copy.deepcopy(config)
        changed["early_stopping_rounds"] += 1
        with pytest.raises(ValueError, match="provenance differs"):
            tune.run_tuning(changed, *args[1:], resume=True)
        source = directory / "model_feature_eligibility.csv"
        source.write_text(source.read_text() + "\n")
        with pytest.raises(ValueError, match="provenance differs"):
            tune.run_tuning(*args, resume=True)
    finally:
        mlflow.end_run()


def test_failed_trial_is_recorded_and_can_resume(small_study, monkeypatch):
    config, directory = small_study
    tracking_uri = f"sqlite:///{directory / 'failed-tracking.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment_id = client.create_experiment("failure-test", artifact_location=(directory / "artifacts").as_uri())
    args = (config, directory, 1, tracking_uri, "failure-test", "recoverable-study", directory / "optuna.db", 1)
    original = tune.fit_window
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic training failure")
    monkeypatch.setattr(tune, "fit_window", fail)
    try:
        with pytest.raises(RuntimeError, match="Synthetic training failure"):
            tune.run_tuning(*args)
        runs = client.search_runs([experiment_id], "tags.stage = 'hyperparameter_tuning_trial'")
        assert len(runs) == 1
        assert runs[0].info.status == "FAILED" and runs[0].data.tags["optuna_trial_state"] == "FAIL"
        assert len(client.search_logged_models(experiment_ids=[experiment_id])) == 0
        monkeypatch.setattr(tune, "fit_window", original)
        result = tune.run_tuning(*args[:-1], 2, resume=True)
        parent = client.get_run(result["parent_run_id"])
        assert parent.data.metrics["trials_fail"] == 1
        assert parent.data.metrics["trials_complete"] == 1
        assert len(client.search_logged_models(experiment_ids=[experiment_id])) == 1
    finally:
        mlflow.end_run()


def test_second_worker_cannot_take_over_active_database(tmp_path):
    with tune.study_lock(tmp_path / "study.db"):
        with pytest.raises(ValueError, match="Another tuning process"):
            with tune.study_lock(tmp_path / "study.db"):
                pytest.fail("Concurrent access must be rejected")
