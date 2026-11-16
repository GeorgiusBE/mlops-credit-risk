"""Check frozen-model scoring, test-only reads, and MLflow model linkage."""

from importlib.metadata import version
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import pytest

from feature_selection import train_baseline as baseline
from model_evaluation import evaluate_test as evaluation


@pytest.fixture
def frozen_refit(tmp_path):
    tracking = f"sqlite:///{tmp_path / 'tracking.db'}"
    mlflow.set_tracking_uri(tracking)
    client = mlflow.MlflowClient()
    experiment = client.create_experiment("test-evaluation", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test-evaluation")
    frame = pd.DataFrame({
        "WEEK_NUM": np.repeat(np.arange(6), 40), "target": [0, 1] * 120,
        "amount": np.tile(np.arange(40), 6).astype(float),
        "industry": ["A", "B"] * 120,
    })
    train = frame.loc[frame["WEEK_NUM"] < 3].copy()
    train["industry"] = train["industry"].astype("category")
    features = ["amount", "industry"]
    model = lgb.LGBMClassifier(n_estimators=5, num_leaves=3, min_child_samples=2, n_jobs=1, verbosity=-1)
    model.fit(train[features], train["target"])
    # Any scoring read outside weeks 3–4 would encounter invalid labels.
    frame.loc[~frame["WEEK_NUM"].isin([3, 4]), "target"] = 999
    frame.loc[[120, 121, 160], "industry"] = "test_only"
    frame.to_parquet(tmp_path / "train_base_engineered.parquet", index=False)
    pd.DataFrame({"column_name": features, "role": ["eligible"] * 2}).to_csv(
        tmp_path / "model_feature_eligibility.csv", index=False,
    )
    context = {
        "configuration": {"split": {"train_start": 0, "train_end": 1, "validation_end": 2, "test_start": 3, "test_end": 4}},
        "features": features,
        "fingerprints": {
            "dataset_sha256": baseline.file_sha256(tmp_path / "train_base_engineered.parquet"),
            "feature_inventory_sha256": baseline.file_sha256(tmp_path / "model_feature_eligibility.csv"),
        },
        "runtime_versions": {name: version(name) for name in ("lightgbm", "pandas", "numpy", "scipy", "scikit-learn")},
    }
    with mlflow.start_run(experiment_id=experiment, run_name="study") as parent:
        with mlflow.start_run(run_name="selected-model", nested=True, tags={
            "stage": "final_refit", "feature_set": "without_deposit_history", "result_status": "refitted_test_not_evaluated",
        }) as source:
            mlflow.log_dict(context, "provenance.json")
            mlflow.log_dict({"features": features, "categorical_features": ["industry"]}, "features.json")
            mlflow.log_dict({"amount": "float64", "industry": "category"}, "feature_dtypes.json")
            mlflow.log_dict({"industry": ["A", "B"]}, "categorical_levels.json")
            mlflow.log_dict({"train": [0, 1, 2], "held_out_test_range": [3, 4], "test_evaluated": False}, "split_weeks.json")
            info = mlflow.lightgbm.log_model(model.booster_, name="model", signature=False,
                                           pip_requirements=[f"lightgbm=={version('lightgbm')}"])
        mlflow.log_dict({"run_id": source.info.run_id, "model_uri": info.model_uri,
                         "source_trial_run_id": "frozen-trial-reference", "n_estimators": 5}, "selected_model.json")
    return {"directory": tmp_path, "tracking": tracking, "source_id": source.info.run_id,
            "model_id": info.model_id, "experiment_id": experiment, "client": client}


def call(fixture, **kwargs):
    return evaluation.evaluate_saved_model(fixture["source_id"], fixture["directory"], fixture["tracking"],
                                           threads=1, lock_dir=fixture["directory"] / "locks", **kwargs)


def test_scores_only_test_rows_without_retraining_and_publishes_model_metrics(frozen_refit, monkeypatch):
    fixture = frozen_refit
    def unexpected(*args, **kwargs):
        pytest.fail("Evaluation must never fit or retrain a model")
    monkeypatch.setattr(lgb.LGBMClassifier, "fit", unexpected)
    monkeypatch.setattr(lgb, "train", unexpected)
    frozen = evaluation.frozen_model(fixture["client"], fixture["source_id"])
    frame, unknown = evaluation.load_test_frame(fixture["directory"], frozen, 1)
    assert frame["WEEK_NUM"].unique().tolist() == [3, 4]
    assert set(frame["target"].unique()) == {0, 1}
    assert frame["industry"].cat.categories.tolist() == ["A", "B"]
    assert unknown == {"industry": 3} and frame["industry"].isna().sum() == 3
    result = call(fixture)
    metrics = result["metrics"]
    assert not result["reused"]
    assert metrics["test_cases"] == 80 and metrics["test_defaults"] == 40
    assert 0 <= metrics["test_roc_auc"] <= 1 and np.isfinite(metrics["test_gini_stability"])
    client = fixture["client"]
    run = client.get_run(result["evaluation_run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["mlflow.parentRunId"] == fixture["source_id"]
    assert run.data.params["retraining"] == "False"
    source = client.get_run(fixture["source_id"])
    assert source.info.status == "FINISHED" and source.data.tags["result_status"] == "test_evaluated"
    assert source.data.metrics["test_roc_auc"] == metrics["test_roc_auc"]
    assert source.data.tags["test_evaluation_run_id"] == run.info.run_id
    assert any(metric.key == "test_roc_auc" for metric in client.get_logged_model(fixture["model_id"]).metrics)
    assert len(client.search_logged_models(experiment_ids=[fixture["experiment_id"]])) == 1
    weekly = evaluation.read_json(client, run.info.run_id, "test_weekly_metrics.json")
    assert len(weekly["data"]) == 2
    monkeypatch.setattr(evaluation, "load_test_frame", unexpected)
    repeated = call(fixture)
    assert repeated["reused"] and repeated["evaluation_run_id"] == result["evaluation_run_id"]


def test_plan_does_not_read_test_rows_or_log_a_run(frozen_refit, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A plan must not read held-out rows")
    monkeypatch.setattr(evaluation, "load_test_frame", unexpected)
    before = len(frozen_refit["client"].search_runs([frozen_refit["experiment_id"]]))
    plan = call(frozen_refit, dry_run=True)
    assert plan["test_range"] == [3, 4] and plan["retraining"] is False
    assert len(frozen_refit["client"].search_runs([frozen_refit["experiment_id"]])) == before
    assert not (frozen_refit["directory"] / "locks").exists()


@pytest.mark.parametrize("change", ["dataset", "inventory", "overlap"])
def test_reject_replaced_data_and_overlapping_test_before_scoring(frozen_refit, monkeypatch, change):
    fixture = frozen_refit
    if change == "dataset":
        path = fixture["directory"] / "train_base_engineered.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "amount"] += 1
        frame.to_parquet(path, index=False)
    elif change == "inventory":
        path = fixture["directory"] / "model_feature_eligibility.csv"
        path.write_text(path.read_text() + "\n")
    else:
        original = evaluation.read_json
        def altered(client, run_id, filename):
            data = original(client, run_id, filename)
            if filename == "split_weeks.json":
                data["held_out_test_range"] = [2, 4]
            return data
        monkeypatch.setattr(evaluation, "read_json", altered)
    with pytest.raises(ValueError, match="differs|overlaps"):
        call(fixture)
    assert not fixture["client"].search_runs([fixture["experiment_id"]], "tags.stage = 'test_evaluation'")
