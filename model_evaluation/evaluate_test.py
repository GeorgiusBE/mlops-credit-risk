"""Evaluate an explicitly selected saved refit using its frozen test boundary."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import duckdb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_selection import train_baseline as baseline


def read_json(client, run_id, filename):
    with TemporaryDirectory() as directory:
        return json.loads(Path(client.download_artifacts(run_id, filename, directory)).read_text())


def frozen_model(client, model_run_id):
    """Resolve the selected model from recorded artifacts, never current config."""
    source = client.get_run(model_run_id)
    if source.info.status != "FINISHED" or source.data.tags.get("stage") != "final_refit":
        raise ValueError("Select a finished final_refit run.")
    parent_id = source.data.tags["mlflow.parentRunId"]
    selection = read_json(client, parent_id, "selected_model.json")
    if selection["run_id"] != model_run_id:
        raise ValueError("The selected model manifest does not identify this refit.")
    model_id = selection["model_uri"].removeprefix("models:/")
    logged = client.get_logged_model(model_id)
    if logged.source_run_id != model_run_id or logged.status.name != "READY":
        raise ValueError("The saved model must be ready and belong to the selected refit.")
    inputs = read_model_input_metadata(client, model_run_id)
    splits = read_json(client, model_run_id, "split_weeks.json")
    split = inputs["context"]["configuration"]["split"]
    start, end = splits["held_out_test_range"]
    train = splits["train"]
    if (not train or not all(type(week) is int for week in [*train, start, end])
            or max(train) >= start or start > end
            or [start, end] != [split["test_start"], split["test_end"]]
            or max(train) != split["validation_end"]):
        raise ValueError("The recorded test boundary overlaps training or differs from the frozen configuration.")
    return {**inputs, "source": source, "parent_run_id": parent_id, "selection": selection,
            "model_uri": selection["model_uri"], "model_id": model_id, "test_range": [start, end]}


def read_model_input_metadata(client, model_run_id):
    """Read the input contract saved with a model's source run."""
    context = read_json(client, model_run_id, "provenance.json")
    membership = read_json(client, model_run_id, "features.json")
    dtypes = read_json(client, model_run_id, "feature_dtypes.json")
    levels = read_json(client, model_run_id, "categorical_levels.json")
    features, categorical = membership["features"], membership["categorical_features"]
    if (not features or len(features) != len(set(features)) or baseline.RESERVED_COLUMNS.intersection(features)
            or features != context["features"] or set(dtypes) != set(features)
            or set(levels) != set(categorical) or not set(categorical) <= set(features)
            or {name for name, dtype in dtypes.items() if dtype == "category"} != set(categorical)):
        raise ValueError("Saved feature order, dtypes, and categorical metadata are inconsistent.")
    return {"context": context, "features": features, "categorical": categorical, "dtypes": dtypes, "levels": levels}


def check_inputs(directory, frozen):
    """Reject replaced data and incompatible scoring environments before test access."""
    fingerprints = {
        "dataset_sha256": baseline.file_sha256(directory / "train_base_engineered.parquet"),
        "feature_inventory_sha256": baseline.file_sha256(directory / "model_feature_eligibility.csv"),
    }
    if fingerprints != frozen["context"]["fingerprints"]:
        raise ValueError("Dataset or feature inventory differs from the saved model's training inputs.")
    check_scoring_runtime(frozen)
    return fingerprints


def check_scoring_runtime(frozen):
    expected = frozen["context"]["runtime_versions"]
    for name in ("lightgbm", "pandas", "numpy", "scipy", "scikit-learn"):
        if version(name) != expected[name]:
            raise ValueError(f"Scoring package {name} differs from the saved model's environment.")


def load_frozen_booster(frozen):
    model = mlflow.lightgbm.load_model(frozen["model_uri"])
    if (model.feature_name() != frozen["features"] or
            (model.pandas_categorical or []) != [frozen["levels"][name] for name in frozen["features"]
                                                if name in frozen["categorical"]]):
        raise ValueError("Saved model inputs do not match the feature/category artifacts.")
    return model


def predict_probabilities(model, frame, threads):
    probabilities = model.predict(frame, num_iteration=-1, num_threads=threads)
    if (probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("The saved binary model did not produce valid default probabilities.")
    return probabilities


def load_test_frame(directory, frozen, threads):
    """Read test rows only; apply saved dtypes and training categorical levels."""
    start, end = frozen["test_range"]
    features = frozen["features"]
    with duckdb.connect(config={"threads": threads}) as connection:
        relation = connection.read_parquet(str(directory / "train_base_engineered.parquet")).select(
            *(duckdb.ColumnExpression(name) for name in ["WEEK_NUM", "target", *features])
        )
        frame = baseline.read_week_range(relation, [start, end])
    if sorted(frame["WEEK_NUM"].unique().tolist()) != list(range(start, end + 1)):
        raise ValueError("Test data must contain every week in the frozen test range.")
    if frame["target"].isna().any() or set(frame["target"].unique()) != {0, 1}:
        raise ValueError("Test labels must contain both binary classes and no missing targets.")
    prepared, unknown = prepare_model_inputs(frame, frozen)
    frame[features] = prepared
    return frame, unknown


def prepare_model_inputs(frame, frozen):
    """Order and cast features with the training vocabulary for each category."""
    features = frame[frozen["features"]].copy()
    unknown = {}
    for name in frozen["features"]:
        if name in frozen["categorical"]:
            levels = frozen["levels"][name]
            known = features[name].isin(levels)
            unknown[name] = int((features[name].notna() & ~known).sum())
            features[name] = features[name].where(known).astype(pd.CategoricalDtype(categories=levels))
        else:
            features[name] = features[name].astype(frozen["dtypes"][name])
    return features, unknown


def publish_result(client, frozen, result):
    """Make test scores discoverable on the source refit without reopening it."""
    source_id = frozen["source"].info.run_id
    for name, value in result["metrics"].items():
        client.log_metric(source_id, name, value)
    client.set_tag(source_id, "test_evaluation_run_id", result["evaluation_run_id"])
    client.set_tag(source_id, "result_status", "test_evaluated")
    client.set_tag(frozen["parent_run_id"], "test_evaluation_run_id", result["evaluation_run_id"])
    client.set_tag(frozen["parent_run_id"], "result_status", "test_evaluated")


@contextmanager
def evaluation_lock(directory):
    """Serialize evaluations on this checkout, including repeat detection."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "test-evaluation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A test evaluation is already running in this checkout.") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def evaluate_saved_model(model_run_id, processed_dir, tracking_uri, threads=2, *, dry_run=False,
                         run_name=None, lock_dir=None):
    if threads < 1:
        raise ValueError("threads must be positive.")
    directory = Path(processed_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    frozen = frozen_model(client, model_run_id)
    fingerprints = check_inputs(directory, frozen)
    plan = {"model_run_id": model_run_id, "model_uri": frozen["model_uri"],
            "source_trial_run_id": frozen["selection"]["source_trial_run_id"],
            "features": frozen["features"], "test_range": frozen["test_range"],
            "n_estimators": frozen["selection"]["n_estimators"], "retraining": False}
    if dry_run:
        return plan
    with evaluation_lock(Path(lock_dir) if lock_dir else baseline.PROJECT_ROOT / ".mlflow"):
        previous = client.search_runs(
            [frozen["source"].info.experiment_id],
            f"tags.stage = 'test_evaluation' AND tags.source_model_run_id = '{model_run_id}' AND attributes.status = 'FINISHED'",
        )
        if previous:
            if len(previous) != 1:
                raise ValueError("Multiple completed evaluations exist; inspect their provenance before continuing.")
            result = read_json(client, previous[0].info.run_id, "evaluation_summary.json")
            publish_result(client, frozen, result)
            return {**result, "reused": True}
        model = load_frozen_booster(frozen)
        model_hash = hashlib.sha256(model.model_to_string().encode()).hexdigest()
        with mlflow.start_run(experiment_id=frozen["source"].info.experiment_id,
                              run_name=run_name or "selected-model-test", tags={
                                  "stage": "test_evaluation", "mlflow.parentRunId": model_run_id,
                                  "source_model_run_id": model_run_id, "source_model_uri": frozen["model_uri"],
                                  "source_trial_run_id": plan["source_trial_run_id"],
                                  "feature_set": frozen["source"].data.tags["feature_set"],
                                  "result_status": "evaluating_frozen_model",
                              }) as run:
            started = time.monotonic()
            mlflow.log_params({"feature_count": len(frozen["features"]), "test_week_start": frozen["test_range"][0],
                               "test_week_end": frozen["test_range"][1], "prediction_threads": threads,
                               "boosting_rounds": model.current_iteration(), "retraining": False,
                               "model_sha256": model_hash, **fingerprints})
            mlflow.log_dict({**plan, "training_provenance": frozen["context"],
                             "evaluator_sha256": baseline.file_sha256(Path(__file__)),
                             "metric_code_sha256": baseline.file_sha256(Path(baseline.__file__)),
                             "evaluation_runtime": {name: version(name) for name in (
                                 "mlflow", "lightgbm", "duckdb", "pandas", "numpy", "scipy", "scikit-learn",
                             )}}, "evaluation_provenance.json")
            mlflow.log_dict({"features": frozen["features"], "categorical_features": frozen["categorical"]}, "features.json")
            mlflow.log_dict(frozen["levels"], "categorical_levels.json")
            mlflow.log_artifact(str(Path(__file__).resolve()), "provenance")
            mlflow.log_artifact(str(Path(baseline.__file__).resolve()), "provenance")
            frame, unknown = load_test_frame(directory, frozen, threads)
            check_inputs(directory, frozen)
            probabilities = predict_probabilities(model, frame[frozen["features"]], threads)
            metrics, weekly = baseline.evaluate(frame["WEEK_NUM"], frame["target"], probabilities)
            scores = {f"test_{name}": float(value) for name, value in metrics.items()}
            scores.update(test_cases=float(len(frame)), test_defaults=float(frame["target"].sum()))
            mlflow.log_metrics(scores, model_id=frozen["model_id"], synchronous=True)
            mlflow.log_metric("duration_seconds", time.monotonic() - started)
            for row in weekly.to_dict("records"):
                mlflow.log_metrics({f"test_weekly_{name}": float(row[name]) for name in (
                    "cases", "defaults", "roc_auc", "gini",
                ) if pd.notna(row[name])}, step=int(row["WEEK_NUM"]))
            mlflow.log_table(weekly, "test_weekly_metrics.json")
            mlflow.log_dict(scores, "test_metrics.json")
            mlflow.log_dict(unknown, "unseen_category_counts.json")
            result = {"evaluation_run_id": run.info.run_id, "model_run_id": model_run_id,
                      "model_uri": frozen["model_uri"], "test_range": frozen["test_range"], "metrics": scores}
            mlflow.log_dict(result, "evaluation_summary.json")
            mlflow.set_tag("result_status", "test_evaluated")
        publish_result(client, frozen, result)
        return {**result, "reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-id", required=True, help="Explicit finished selected-model refit run ID.")
    parser.add_argument("--processed-dir", type=Path, default=baseline.PROJECT_ROOT / "dataset" / "processed")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true", help="Inspect frozen artifacts and hashes without reading test rows or logging a run.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    result = evaluate_saved_model(args.model_run_id, args.processed_dir, args.tracking_uri, args.threads,
                                  dry_run=args.dry_run, run_name=args.run_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
