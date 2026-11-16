"""Tune LightGBM on temporal windows, then save one fixed-round final refit."""

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import duckdb
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_selection import train_baseline as baseline


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "lightgbm.json"
DEFAULT_STORAGE = baseline.PROJECT_ROOT / ".optuna" / "studies.db"
PRIMARY_METRIC = "mean_validation_gini_stability"
TUNABLE = {
    "learning_rate", "num_leaves", "max_depth", "min_child_samples",
    "colsample_bytree", "reg_alpha", "reg_lambda",
}


def validate_config(config):
    """Reject invalid windows, unsafe search parameters, and unusable controls."""
    split = config["split"]
    if not (type(split.get("test_start")) is int and
            split["validation_end"] + 1 == split["test_start"] <= split["test_end"]):
        raise ValueError("The final test must start immediately after development validation.")
    windows = config["windows"]
    if not isinstance(windows, list) or len(windows) < 2:
        raise ValueError("Use at least two chronological validation windows.")
    previous_end = split["train_start"] - 1
    names = set()
    for window in windows:
        name = window["name"]
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("Window names must be unique and contain only letters, digits, or underscores.")
        names.add(name)
        if not all(type(window[key]) is int for key in ("train_end", "validation_end")) or not (
            split["train_start"] <= window["train_end"] < window["validation_end"] < split["test_start"]
        ):
            raise ValueError("Every training/validation window must precede the final test.")
        if window["train_end"] < previous_end:
            raise ValueError("Validation windows must be chronological and must not overlap.")
        previous_end = window["validation_end"]
    if previous_end != split["validation_end"]:
        raise ValueError("The last validation window must end at the development boundary.")
    if type(config["sampler_seed"]) is not int or config["sampler_seed"] < 0:
        raise ValueError("sampler_seed must be a nonnegative integer.")
    if config["refit_rounds_rule"] != "median_best_iteration":
        raise ValueError("Expected median_best_iteration as the final refit rule.")
    model = config["model"]
    if type(model["n_estimators"]) is not int or model["n_estimators"] < 1:
        raise ValueError("n_estimators must be a positive boosting-round cap.")
    space = config["search_space"]
    if not space or set(space) - TUNABLE:
        raise ValueError(f"Search parameters must be chosen from {sorted(TUNABLE)}.")
    for name, specification in space.items():
        kind = specification["type"]
        current = model[name]
        if kind == "categorical":
            choices = specification["choices"]
            valid = bool(choices) and current in choices
        elif kind in ("int", "float"):
            low, high = specification["low"], specification["high"]
            valid = (all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                         for value in (low, high, current)) and low <= current <= high)
            if kind == "int":
                valid = valid and all(type(value) is int for value in (low, high, current))
            if specification.get("log", False):
                valid = valid and low > 0
        else:
            valid = False
        if not valid:
            raise ValueError(f"Invalid search space or current-parameter control for {name}.")
    return config


def selected_features(config, inventory_path):
    """Apply the recorded human feature decision while preserving inventory order."""
    eligible = baseline.eligible_features(inventory_path)
    decision = config["feature_selection"]
    exclusions = decision["excluded_features"]
    if (decision.get("selection_method") != "human" or not isinstance(exclusions, list) or
            not exclusions or not all(isinstance(name, str) for name in exclusions) or
            len(exclusions) != len(set(exclusions)) or not set(exclusions) <= set(eligible)):
        raise ValueError("Feature exclusions must be an explicit human choice of distinct eligible inputs.")
    features = [name for name in eligible if name not in exclusions]
    if not features or len(features) != decision["expected_feature_count"]:
        raise ValueError("Feature count differs from the recorded decision; review the feature inventory.")
    return features


def current_window_config(config, window):
    result = copy.deepcopy(config)
    result["split"].update({name: window[name] for name in ("train_end", "validation_end")})
    return result


def provenance(config, features, processed_dir, threads, tracking_uri, experiment_id):
    """Describe all inputs that must match before extending an existing study."""
    runtime = {name: version(name) for name in (
        "optuna", "mlflow", "lightgbm", "duckdb", "pandas", "numpy", "scipy", "scikit-learn",
    )}
    runtime["python"] = sys.version
    sources = [Path(__file__).resolve(), Path(baseline.__file__).resolve(),
               baseline.PROJECT_ROOT / "pyproject.toml", baseline.PROJECT_ROOT / "poetry.lock"]
    return {
        "configuration": config, "features": features, "threads": threads,
        "tracking_uri": tracking_uri.rstrip("/"), "experiment_id": experiment_id,
        "fingerprints": {
            "dataset_sha256": baseline.file_sha256(processed_dir / "train_base_engineered.parquet"),
            "feature_inventory_sha256": baseline.file_sha256(processed_dir / "model_feature_eligibility.csv"),
        },
        "runtime_versions": runtime,
        "source_sha256": {str(path.relative_to(baseline.PROJECT_ROOT)): baseline.file_sha256(path) for path in sources},
    }


def check_fingerprints(data, expected):
    if data.fingerprints != expected:
        raise ValueError("Training inputs changed during the study; start a new study after reviewing the change.")


def fit_window(data, model_params, patience, threads):
    """Fit one window with AUC early stopping; return scores, never a saved model."""
    features = data.features
    model = lgb.LGBMClassifier(**{**model_params, "n_jobs": threads})
    started = time.monotonic()
    model.fit(
        data.train[features], data.train["target"],
        eval_X=data.validation[features], eval_y=data.validation["target"],
        callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
    )
    probabilities = model.predict_proba(data.validation[features], num_iteration=model.best_iteration_)[:, 1]
    metrics, weekly = baseline.evaluate(data.validation["WEEK_NUM"], data.validation["target"], probabilities)
    return {
        "metrics": metrics, "weekly": weekly, "best_iteration": int(model.best_iteration_),
        "duration_seconds": time.monotonic() - started,
    }


def summarize_results(results):
    """Give each temporal window equal weight and keep worst-window behaviour."""
    if not results:
        raise ValueError("No temporal validation results were supplied.")
    summary = {
        f"mean_validation_{name}": float(np.mean([result["metrics"][name] for result in results]))
        for name in results[0]["metrics"]
    }
    summary["worst_validation_gini_stability"] = min(result["metrics"]["gini_stability"] for result in results)
    if not all(math.isfinite(value) for value in summary.values()):
        raise ValueError("Temporal validation produced nonfinite metrics.")
    return summary


def refit_rounds(best_iterations):
    """Round the median best iteration to the nearest positive integer, halves up."""
    if not best_iterations or any(type(value) is not int or value < 1 for value in best_iterations):
        raise ValueError("Refit requires positive best iterations from every validation window.")
    return int(math.floor(statistics.median(best_iterations) + 0.5))


def objective(trial, config, features, processed_dir, threads, fingerprints):
    """Record one child per parameter combination and release each window's data."""
    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True) as run:
        started = time.monotonic()
        mlflow.set_tags({
            "stage": "hyperparameter_tuning_trial", "model_family": "lightgbm",
            "feature_set": config["feature_selection"]["name"],
            "optuna_trial_number": trial.number, "trial_role": trial.user_attrs.get("role", "search"),
        })
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        try:
            sampled = {}
            for name, specification in config["search_space"].items():
                arguments = {key: value for key, value in specification.items() if key != "type"}
                sampled[name] = getattr(trial, f"suggest_{specification['type']}")(name, **arguments)
            model_params = {**config["model"], **sampled}
            mlflow.log_params({**model_params, "n_jobs": threads, "feature_count": len(features),
                               "early_stopping_rounds": config["early_stopping_rounds"]})
            mlflow.log_dict(model_params, "model_parameters.json")
            mlflow.log_dict({"features": features}, "features.json")
            results, rows = [], []
            for window in config["windows"]:
                print(f"Trial {trial.number}: {window['name']}", flush=True)
                current = current_window_config(config, window)
                data = baseline.load_training_data(processed_dir, current, threads, features=features)
                check_fingerprints(data, fingerprints)
                result = fit_window(data, model_params, config["early_stopping_rounds"], threads)
                name = window["name"]
                metrics = {**result["metrics"], "best_iteration": result["best_iteration"],
                           "duration_seconds": result["duration_seconds"],
                           "train_cases": len(data.train), "validation_cases": len(data.validation),
                           "reached_round_cap": int(result["best_iteration"] >= model_params["n_estimators"])}
                mlflow.log_metrics({f"{name}_{key}": value for key, value in metrics.items()})
                mlflow.log_dict(data.splits, f"windows/{name}/split_weeks.json")
                mlflow.log_dict({column: data.train[column].cat.categories.tolist() for column in data.categorical},
                                f"windows/{name}/categorical_levels.json")
                mlflow.log_table(result["weekly"], f"windows/{name}/weekly_metrics.json")
                results.append(result)
                rows.append({"window": name, **metrics})
                del data
            summary = summarize_results(results)
            mlflow.log_metrics(summary)
            mlflow.log_table(pd.DataFrame(rows), "validation_by_window.json")
            trial.set_user_attr("best_iterations", [result["best_iteration"] for result in results])
            trial.set_user_attr("summary_metrics", summary)
            mlflow.set_tag("optuna_trial_state", "COMPLETE")
            return summary[PRIMARY_METRIC]
        except BaseException as error:
            mlflow.set_tag("optuna_trial_state", "FAIL")
            mlflow.set_tag("failure_type", type(error).__name__)
            raise
        finally:
            mlflow.log_metric("duration_seconds", time.monotonic() - started)


def load_refit_frame(processed_dir, config, features, threads, fingerprints):
    """Read development rows only and learn categories again on all those rows."""
    split = config["split"]
    with duckdb.connect(config={"threads": threads}) as connection:
        relation = connection.read_parquet(str(processed_dir / "train_base_engineered.parquet")).select(
            *(duckdb.ColumnExpression(name) for name in ["WEEK_NUM", "target", *features])
        )
        frame = baseline.read_week_range(relation, [split["train_start"], split["validation_end"]])
        categorical = [name for name, dtype in zip(relation.columns, map(str, relation.types))
                       if name in features and dtype == "VARCHAR"]
    actual = {
        "dataset_sha256": baseline.file_sha256(processed_dir / "train_base_engineered.parquet"),
        "feature_inventory_sha256": baseline.file_sha256(processed_dir / "model_feature_eligibility.csv"),
    }
    if actual != fingerprints:
        raise ValueError("Training inputs changed before final refit; start a new study.")
    if frame["target"].isna().any() or set(frame["target"].unique()) != {0, 1}:
        raise ValueError("Development rows must have both binary classes and no missing labels.")
    for column in categorical:
        frame[column] = frame[column].astype("category")
    return frame, categorical


def refit_and_log(study, config, features, processed_dir, threads, context):
    """Save a fresh model trained on weeks 0–77 without validation or test access."""
    best = study.best_trial
    iterations = best.user_attrs["best_iterations"]
    if len(iterations) != len(config["windows"]):
        raise ValueError("The selected trial is missing a window's best iteration.")
    rounds = refit_rounds(iterations)
    parameters = {**config["model"], **best.params, "n_estimators": rounds, "n_jobs": threads}
    with mlflow.start_run(run_name="selected-model", nested=True) as run:
        started = time.monotonic()
        mlflow.set_tags({
            "stage": "final_refit", "model_family": "lightgbm", "feature_set": config["feature_selection"]["name"],
            "source_trial_run_id": best.user_attrs["mlflow_run_id"], "optuna_study_name": study.study_name,
            "result_status": "refitted_test_not_evaluated",
        })
        mlflow.log_params({**parameters, "feature_count": len(features), "selected_trial_number": best.number,
                           "train_week_start": config["split"]["train_start"],
                           "train_week_end": config["split"]["validation_end"],
                           "early_stopping": False, "refit_rounds_rule": config["refit_rounds_rule"]})
        mlflow.log_dict({"window_best_iterations": iterations, "refit_rounds": rounds,
                         "rule": config["refit_rounds_rule"]}, "refit_rounds.json")
        frame, categorical = load_refit_frame(processed_dir, config, features, threads, context["fingerprints"])
        mlflow.log_param("train_cases", len(frame))
        mlflow.log_dict({"train": sorted(map(int, frame["WEEK_NUM"].unique())),
                         "held_out_test_range": [config["split"]["test_start"], config["split"]["test_end"]],
                         "test_evaluated": False}, "split_weeks.json")
        mlflow.log_dict({"features": features, "categorical_features": categorical}, "features.json")
        mlflow.log_dict({name: str(frame[name].dtype) for name in features}, "feature_dtypes.json")
        mlflow.log_dict({name: frame[name].cat.categories.tolist() for name in categorical}, "categorical_levels.json")
        mlflow.log_dict(context, "provenance.json")
        mlflow.log_dict(parameters, "model_parameters.json")
        model = lgb.LGBMClassifier(**parameters)
        model.fit(frame[features], frame["target"])
        info = mlflow.lightgbm.log_model(
            model.booster_, name="model", signature=False,
            pip_requirements=[f"{name}=={context['runtime_versions'][name]}"
                              for name in ("lightgbm", "pandas", "numpy", "scipy", "scikit-learn")],
        )
        mlflow.log_metrics({"duration_seconds": time.monotonic() - started, "actual_boosting_rounds": model.n_iter_})
        # Trial validation metrics stay on the trial; they do not describe this refit's performance.
        return {"run_id": run.info.run_id, "model_uri": info.model_uri,
                "selected_trial_number": best.number, "n_estimators": rounds,
                "source_trial_run_id": best.user_attrs["mlflow_run_id"]}


def log_study_summary(study):
    """Export all attempts and a compact comparison with direct MLflow run links."""
    trials = study.get_trials(deepcopy=False)
    rows = [{"trial_number": trial.number, "state": trial.state.name, "objective": trial.value,
             "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"), **trial.params,
             **trial.user_attrs.get("summary_metrics", {})} for trial in trials]
    if rows:
        mlflow.log_table(pd.DataFrame(rows), "trial_comparison.json")
    mlflow.log_metrics({f"trials_{state.name.lower()}": sum(trial.state == state for trial in trials)
                        for state in optuna.trial.TrialState})
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if complete:
        best = study.best_trial
        mlflow.log_metric("best_mean_validation_gini_stability", best.value)
        mlflow.set_tags({"best_trial_number": best.number, "best_trial_run_id": best.user_attrs["mlflow_run_id"]})
        mlflow.log_dict({"trial_number": best.number, "parameters": best.params,
                         "validation_metrics": best.user_attrs["summary_metrics"],
                         "best_iterations": best.user_attrs["best_iterations"],
                         "mlflow_run_id": best.user_attrs["mlflow_run_id"]}, "selected_trial.json")


@contextmanager
def study_lock(storage_path):
    """Prevent a second worker from treating an active trial as interrupted."""
    path = Path(storage_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another tuning process is using this Optuna database; wait for it to finish.") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_tuning(config, processed_dir, threads, tracking_uri, experiment_name, study_name,
               storage_path, n_trials, *, resume=False, timeout=None, run_name=None):
    """Hold an exclusive local worker lock for the entire study/refit invocation."""
    if threads < 1 or n_trials < 1 or (timeout is not None and (not math.isfinite(timeout) or timeout <= 0)):
        raise ValueError("Threads, total trials, and any timeout must be positive.")
    with study_lock(storage_path):
        return _run_tuning(config, processed_dir, threads, tracking_uri, experiment_name, study_name,
                           storage_path, n_trials, resume=resume, timeout=timeout, run_name=run_name)


def _run_tuning(config, processed_dir, threads, tracking_uri, experiment_name, study_name,
                storage_path, n_trials, *, resume=False, timeout=None, run_name=None):
    """Run a sequential, resumable study; n_trials is a total attempt budget."""
    validate_config(config)
    processed_dir, storage_path = Path(processed_dir), Path(storage_path).resolve()
    features = selected_features(config, processed_dir / "model_feature_eligibility.csv")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    context = provenance(config, features, processed_dir, threads, tracking_uri, experiment.experiment_id)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=config["sampler_seed"])
    if resume:
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        if study.user_attrs.get("context") != context or not study.user_attrs.get("mlflow_parent_run_id"):
            raise ValueError("Study provenance differs from the current inputs, code, runtime, or configuration; use a new study name.")
    else:
        study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler,
                                    direction="maximize", pruner=optuna.pruners.NopPruner())
        study.set_user_attr("context", context)
        study.enqueue_trial({name: config["model"][name] for name in config["search_space"]},
                            user_attrs={"role": "current_parameters"})
        
    with mlflow.start_run(run_id=study.user_attrs.get("mlflow_parent_run_id"), run_name=run_name or study_name) as parent:
        study.set_user_attr("mlflow_parent_run_id", parent.info.run_id)
        print(f"Optuna study: {study_name}\nMLflow parent run: {parent.info.run_id}", flush=True)
        mlflow.set_tags({"stage": "hyperparameter_tuning", "method": "optuna_tpe", "model_family": "lightgbm",
                         "feature_set": config["feature_selection"]["name"], "optuna_study_name": study_name,
                         "result_status": "tuning", "primary_metric": PRIMARY_METRIC})
        mlflow.log_params({"feature_count": len(features), "window_count": len(config["windows"]),
                           "sampler_seed": config["sampler_seed"], "threads": threads,
                           "refit_rounds_rule": config["refit_rounds_rule"], **context["fingerprints"]})
        mlflow.log_dict(context, "provenance.json")
        mlflow.log_dict(config, "configuration.json")
        mlflow.log_dict({"features": features, **config["feature_selection"]}, "feature_selection.json")
        mlflow.set_tag("optuna_storage_path", str(storage_path))
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=baseline.PROJECT_ROOT,
                                  capture_output=True, text=True, check=False)
        if revision.returncode == 0:
            mlflow.set_tag("mlflow.source.git.commit", revision.stdout.strip())
        for relative in context["source_sha256"]:
            mlflow.log_artifact(str(baseline.PROJECT_ROOT / relative), "provenance")
        mlflow.log_artifact(str(processed_dir / "model_feature_eligibility.csv"), "provenance")
        remaining = max(0, n_trials - len([trial for trial in study.trials
                                         if trial.state != optuna.trial.TrialState.WAITING]))
        mlflow.log_dict({"total_trial_budget": n_trials, "remaining_attempts": remaining,
                         "timeout_seconds": timeout, "resumed": resume}, "invocation.json")
        # A hard interruption can leave RUNNING trials; this workflow has one worker.
        for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            if trial.user_attrs.get("mlflow_run_id"):
                client = mlflow.MlflowClient()
                client.set_tag(trial.user_attrs["mlflow_run_id"], "optuna_trial_state", "FAIL")
                client.set_terminated(trial.user_attrs["mlflow_run_id"], status="KILLED")
        try:
            study.optimize(
                lambda trial: objective(trial, config, features, processed_dir, threads, context["fingerprints"]),
                n_trials=remaining, timeout=timeout, n_jobs=1, gc_after_trial=True,
            )
            log_study_summary(study)
            if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
                raise ValueError("No trial completed; no model can be selected or refitted.")
            best = study.best_trial
            previous = study.user_attrs.get("final_refit")
            if (previous and previous["selected_trial_number"] == best.number and
                    mlflow.MlflowClient().get_run(previous["run_id"]).info.status == "FINISHED"):
                result = previous
            else:
                result = refit_and_log(study, config, features, processed_dir, threads, context)
                study.set_user_attr("final_refit", result)
            mlflow.log_dict(result, "selected_model.json")
            mlflow.set_tags({"result_status": "refitted_test_not_evaluated", "selected_model_run_id": result["run_id"],
                             "selected_model_uri": result["model_uri"]})
        except BaseException:
            mlflow.set_tag("result_status", "interrupted_or_failed")
            log_study_summary(study)
            raise
    return {"study_name": study_name, "parent_run_id": parent.info.run_id, **result}


def main():
    parser = baseline.make_parser(__doc__)
    parser.set_defaults(config=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Inspect features/windows without training, databases, or MLflow.")
    parser.add_argument("--study-name", help="Persistent Optuna study name; required with --resume.")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--n-trials", type=int, default=30, help="Total attempts including the current-parameter control and earlier attempts on resume.")
    parser.add_argument("--timeout", type=float, help="Optimization time budget in seconds for this invocation; refit is additional.")
    parser.add_argument("--resume", action="store_true", help="Extend an existing matching study; never silently combine different inputs.")
    args = parser.parse_args()
    if args.threads < 1 or args.n_trials < 1:
        parser.error("--threads and --n-trials must be positive.")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be a positive finite number of seconds.")
    if args.resume and not args.study_name:
        parser.error("--resume requires --study-name.")
    try:
        config = validate_config(baseline.load_config(args.config))
        features = selected_features(config, args.processed_dir / "model_feature_eligibility.csv")
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"feature_selection": config["feature_selection"], "features": features,
                          "windows": config["windows"], "search_space": config["search_space"],
                          "primary_metric": PRIMARY_METRIC, "n_trials": args.n_trials,
                          "sampler_seed": config["sampler_seed"], "early_stopping_rounds": config["early_stopping_rounds"],
                          "boosting_round_cap": config["model"]["n_estimators"],
                          "refit": {"train_start": config["split"]["train_start"],
                                    "train_end": config["split"]["validation_end"],
                                    "rounds_rule": config["refit_rounds_rule"], "early_stopping": False},
                          "reserved_test": [config["split"]["test_start"], config["split"]["test_end"]],
                          "storage_path": str(args.storage_path)}, indent=2))
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = run_tuning(config, args.processed_dir, args.threads, args.tracking_uri, args.experiment_name,
                        args.study_name or args.run_name or f"lightgbm-optuna-{timestamp}", args.storage_path,
                        args.n_trials, resume=args.resume, timeout=args.timeout, run_name=args.run_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
