"""Train the LightGBM baseline and evaluate it on the validation weeks."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "group_ablation.json"
RESERVED_COLUMNS = {"case_id", "date_decision", "MONTH", "WEEK_NUM", "target"}


@dataclass
class TrainingData:
    """Training and validation frames plus provenance; no held-out test labels."""

    train: pd.DataFrame
    validation: pd.DataFrame
    features: list[str]
    categorical: list[str]
    splits: dict[str, list[int]]
    data_path: Path
    inventory_path: Path
    fingerprints: dict[str, str]


def file_sha256(path):
    """Return a content fingerprint without loading the whole file into memory.

    Record this digest with the run so replacing a dataset at the same path
    does not make different training inputs appear to be the same version.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(weeks, labels, probabilities):
    """Return validation metrics and a DataFrame with one row per week.

    Pass one week number, binary default label, and predicted default
    probability per case in the same order. If weeks and labels are pandas
    Series, their indexes must align.

    The metrics include overall ROC AUC, average precision, log loss, and the
    Home Credit Gini stability score. Weeks with a single target class appear
    in the weekly table with missing AUC and Gini. The stability score uses
    only weeks with both classes and raises ValueError if fewer than two qualify.
    """
    predictions = pd.DataFrame({
        "WEEK_NUM": weeks,
        "target": labels,
        "probability": probabilities,
    })
    rows = []
    for week, group in predictions.groupby("WEEK_NUM", sort=True):
        defaults = int(group["target"].sum())
        auc = (
            roc_auc_score(group["target"], group["probability"])
            if 0 < defaults < len(group)
            else np.nan
        )
        rows.append((week, len(group), defaults, auc, 2 * auc - 1))

    weekly = pd.DataFrame(rows, columns=["WEEK_NUM", "cases", "defaults", "roc_auc", "gini"])
    scored = weekly.dropna(subset=["gini"])
    if len(scored) < 2:
        raise ValueError("The stability score requires at least two weeks with both target classes.")

    # Home Credit competition score. Fit a trend over the scored week numbers.
    slope, intercept = np.polyfit(scored["WEEK_NUM"], scored["gini"], 1)
    residuals = scored["gini"] - (slope * scored["WEEK_NUM"] + intercept)
    mean_gini = float(scored["gini"].mean())
    residual_std = float(residuals.std(ddof=0))
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "mean_weekly_gini": mean_gini,
        "weekly_gini_slope": float(slope),
        "weekly_gini_residual_std": residual_std,
        "gini_stability": mean_gini + 88 * min(0.0, float(slope)) - 0.5 * residual_std,
        "weeks_scored": len(scored),
        "weeks_skipped_single_class": len(weekly) - len(scored),
    }
    return metrics, weekly


def read_week_range(relation, weeks):
    """Load the relation's selected columns for an inclusive week range.

    Pass a nonempty, ascending sequence of week numbers. DuckDB filters on
    WEEK_NUM using the first and last values, then returns a pandas DataFrame.
    """
    return relation.filter(f"WEEK_NUM BETWEEN {weeks[0]} AND {weeks[-1]}").df()


def load_config(path):
    """Read shared model settings and fixed week boundaries from JSON.

    AUC remains the early stopping metric. Every candidate in an ablation study
    uses this same model configuration; feature sets cannot override it.
    """
    config = json.loads(Path(path).read_text())
    if config["model"].get("objective") != "binary" or config["model"].get("metric") != "auc":
        raise ValueError("Expected a binary model with AUC early stopping.")
    patience = config["early_stopping_rounds"]
    if type(patience) is not int or patience < 1:
        raise ValueError("early_stopping_rounds must be a positive integer.")
    split = config["split"]
    boundaries = [split[name] for name in ("train_start", "train_end", "validation_end", "test_end")]
    if not all(type(value) is int for value in boundaries) or not (
        boundaries[0] <= boundaries[1] < boundaries[2] < boundaries[3]
    ):
        raise ValueError("Expected ordered, non-overlapping training, validation, and test week boundaries.")
    test_start = split.get("test_start", split["validation_end"] + 1)
    if type(test_start) is not int or not split["validation_end"] < test_start <= split["test_end"]:
        raise ValueError("test_start must follow validation and not exceed test_end.")
    return config


def make_parser(description):
    """Build the shared command-line interface for baseline and ablation runs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "processed",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON model, split, and feature-group settings.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow endpoint (defaults to MLFLOW_TRACKING_URI or the local server).",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit-risk"),
        help="MLflow experiment grouping the baseline and future feature-selection runs.",
    )
    parser.add_argument("--run-name", help="Display name in MLflow (default: generated timestamped name).")
    return parser


def eligible_features(inventory_path):
    """Return eligible feature names in inventory order, rejecting metadata and labels."""
    inventory = pd.read_csv(inventory_path)
    features = inventory.loc[inventory["role"].eq("eligible"), "column_name"].tolist()
    if not features or len(features) != len(set(features)) or RESERVED_COLUMNS.intersection(features):
        raise ValueError("Expected distinct eligible feature names separate from metadata and target.")
    return features


def load_training_data(processed_dir, config, threads, *, features=None):
    """Load selected training/validation rows once, with training-only categories.

    DuckDB reads the eligible columns and filters before returning pandas frames.
    Query week metadata for all rows to verify the fixed split, but never load
    test labels or test feature values. A caller can select an explicit subset
    of eligible inputs before DuckDB reads feature values. Dataset fingerprints
    identify the original data and inventory, including when selecting a subset.
    """
    processed_dir = Path(processed_dir)
    data_path = processed_dir / "train_base_engineered.parquet"
    inventory_path = processed_dir / "model_feature_eligibility.csv"
    if not data_path.is_file() or not inventory_path.is_file():
        raise FileNotFoundError("Run notebooks 3 and 4 to generate the engineered table and inventory.")

    eligible = eligible_features(inventory_path)
    if features is None:
        features = eligible
    elif not features or len(features) != len(set(features)) or not set(features) <= set(eligible):
        raise ValueError("Selected features must be a nonempty, distinct subset of eligible inputs.")

    with duckdb.connect(config={"threads": threads}) as con:
        source = con.read_parquet(str(data_path))
        relation = source.select(
            *(duckdb.ColumnExpression(name) for name in ["WEEK_NUM", "target", *features])
        )
        column_types = dict(zip(relation.columns, map(str, relation.types)))
        categorical = [name for name in features if column_types[name] == "VARCHAR"]
        weeks = [week for (week,) in relation.project("WEEK_NUM").distinct().order("WEEK_NUM").fetchall()]

        split = config["split"]
        if not weeks or None in weeks or weeks[0] < split["train_start"] or weeks[-1] > split["test_end"]:
            raise ValueError("Observed weeks must be non-null and within the configured split boundaries.")
        train_weeks = [week for week in weeks if week <= split["train_end"]]
        val_weeks = [week for week in weeks if split["train_end"] < week <= split["validation_end"]]
        test_start = split.get("test_start", split["validation_end"] + 1)
        test_weeks = [week for week in weeks if week >= test_start]
        unused_weeks = [week for week in weeks if split["validation_end"] < week < test_start]
        if not train_weeks or not val_weeks or not test_weeks:
            raise ValueError("Each configured split must contain observed weeks.")
        train = read_week_range(relation, train_weeks)
        validation = read_week_range(relation, val_weeks)

    for split, labels in (("Training", train["target"]), ("Validation", validation["target"])):
        if labels.isna().any() or set(labels.unique()) != {0, 1}:
            raise ValueError(f"{split} must have both binary classes and no missing targets.")
    for name in categorical:
        train[name] = train[name].astype("category")
        known = validation[name].isin(train[name].cat.categories)
        validation[name] = validation[name].where(known).astype(train[name].dtype)

    splits = {"train": train_weeks, "validation": val_weeks, "held_out_test": test_weeks}
    if unused_weeks:
        splits["unused_future"] = unused_weeks
    return TrainingData(
        train=train, validation=validation, features=features, categorical=categorical,
        splits=splits,
        data_path=data_path, inventory_path=inventory_path,
        fingerprints={"dataset_sha256": file_sha256(data_path), "feature_inventory_sha256": file_sha256(inventory_path)},
    )


def train_and_log(data, config, features, threads, run_name, *, stage="baseline", excluded_groups=(), nested=False):
    """Fit one feature set, log its metrics/model, and return a compact result.

    The caller selects the MLflow experiment. Nested runs belong to the active
    study run. Reuse the prepared rows and categorical vocabularies; only the
    selected input columns vary. Each candidate uses the same stopping rule,
    though its best iteration may differ. Return no fitted data or model so
    sequential ablations can release their large training allocations.
    """
    if not features or len(features) != len(set(features)) or not set(features) <= set(data.features):
        raise ValueError("Candidate features must be a nonempty, distinct subset of eligible inputs.")
    train, validation = data.train, data.validation
    train_weeks, val_weeks, test_weeks = (data.splits[name] for name in ("train", "validation", "held_out_test"))
    categorical = [name for name in data.categorical if name in features]
    X_train, y_train = train[features], train["target"]
    X_val, y_val = validation[features], validation["target"]
    model = lgb.LGBMClassifier(**{**config["model"], "n_jobs": threads})
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags({"stage": stage, "feature_set": run_name, "model_family": "lightgbm"})
        mlflow.log_params(model.get_params())
        mlflow.log_params({
            "feature_count": len(features),
            "categorical_feature_count": len(categorical),
            "train_week_start": train_weeks[0],
            "train_week_end": train_weeks[-1],
            "validation_week_start": val_weeks[0],
            "validation_week_end": val_weeks[-1],
            "test_week_start": test_weeks[0],
            "test_week_end": test_weeks[-1],
            "train_cases": len(train),
            "validation_cases": len(validation),
            "early_stopping_rounds": config["early_stopping_rounds"],
            "excluded_groups": ",".join(excluded_groups) or "none",
            **data.fingerprints,
        })
        runtime_versions = {
            name: version(name)
            for name in ("mlflow", "lightgbm", "duckdb", "pandas", "numpy", "scipy", "scikit-learn")
        }
        mlflow.log_dict({"python": platform.python_version(), **runtime_versions}, "runtime_versions.json")
        mlflow.log_dict(data.splits, "split_weeks.json")
        mlflow.log_dict(config, "configuration.json")
        mlflow.log_dict({"features": features, "categorical_features": categorical}, "features.json")
        mlflow.log_dict({name: str(X_train[name].dtype) for name in features}, "feature_dtypes.json")
        mlflow.log_dict(
            {name: X_train[name].cat.categories.tolist() for name in categorical},
            "categorical_levels.json",
        )
        mlflow.log_artifact(str(data.inventory_path), artifact_path="provenance")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="provenance")
        for filename in ("pyproject.toml", "poetry.lock"):
            mlflow.log_artifact(str(PROJECT_ROOT / filename), artifact_path="provenance")

        print(f"MLflow run: {run.info.run_id}", flush=True)
        print(f"Training on {len(X_train):,} cases with {len(features)} features.", flush=True)
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(config["early_stopping_rounds"], first_metric_only=True), lgb.log_evaluation(50)],
        )
        probabilities = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        metrics, weekly = evaluate(validation["WEEK_NUM"], y_val, probabilities)
        mlflow.log_metrics({f"validation_{name}": value for name, value in metrics.items()})
        mlflow.log_metric("best_iteration", model.best_iteration_)
        for row in weekly.itertuples(index=False):
            week_metrics = {
                "validation_weekly_cases": row.cases,
                "validation_weekly_defaults": row.defaults,
            }
            if pd.notna(row.gini):
                week_metrics.update({
                    "validation_weekly_gini": row.gini,
                    "validation_weekly_roc_auc": row.roc_auc,
                })
            mlflow.log_metrics(week_metrics, step=int(row.WEEK_NUM))

        # Store the native Booster: its predictions are default probabilities,
        # and its saved categorical vocabularies match those learned on training.
        model_info = mlflow.lightgbm.log_model(
            model.booster_,
            name="model",
            # MLflow's string schema rejects pandas categorical dtype input. Preserve
            # the prepared DataFrame; its dtypes and vocabularies are artifacts.
            signature=False,
            pip_requirements=[
                f"{name}=={runtime_versions[name]}"
                for name in ("lightgbm", "pandas", "numpy", "scipy", "scikit-learn")
            ],
        )

    print(f"Logged model to MLflow: {model_info.model_uri}")
    return {
        "run_id": run.info.run_id, "model_uri": model_info.model_uri,
        "feature_count": len(features), "best_iteration": model.best_iteration_,
        "metrics": metrics, "weekly": weekly,
    }


def main():
    """Train the eligible baseline with the same configuration used by ablations."""
    parser = make_parser(__doc__)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    config = load_config(args.config)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    data = load_training_data(args.processed_dir, config, args.threads)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = train_and_log(data, config, data.features, args.threads, args.run_name or f"baseline-{timestamp}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
