"""Compare named feature-group ablations using a fixed LightGBM configuration."""

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

if __package__:
    from . import train_baseline as baseline
else:
    import train_baseline as baseline


def build_feature_sets(features, config):
    """Resolve ablations to eligible columns, retaining the inventory's order.

    Groups must partition the current eligible inputs: reject unknown columns,
    overlaps, or unassigned features so an inventory change cannot silently
    change the meaning of a study. Require exactly one full baseline. Multiple
    exclusions in one configuration allow later confirmation of combined drops.
    """
    groups = config["feature_groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("feature_groups must be a nonempty mapping.")
    for name, columns in groups.items():
        if not isinstance(columns, list) or not columns or not all(isinstance(column, str) for column in columns):
            raise ValueError(f"Group {name!r} must contain a nonempty list of column names.")
    grouped = [column for columns in groups.values() for column in columns]
    if len(grouped) != len(set(grouped)):
        raise ValueError("Each eligible feature must belong to exactly one group; found overlaps.")
    missing, unknown = set(features) - set(grouped), set(grouped) - set(features)
    if missing or unknown:
        raise ValueError(f"Feature groups do not match eligibility. Unassigned: {sorted(missing)}; unknown/ineligible: {sorted(unknown)}")

    ablations = config["ablations"]
    if not isinstance(ablations, dict) or not ablations:
        raise ValueError("ablations must be a nonempty mapping of names to excluded groups.")
    resolved = {}
    for name, exclusions in ablations.items():
        if not isinstance(exclusions, list) or not all(isinstance(group, str) for group in exclusions):
            raise ValueError(f"Ablation {name!r} must list excluded group names.")
        if len(exclusions) != len(set(exclusions)) or set(exclusions) - set(groups):
            raise ValueError(f"Unknown or repeated excluded groups for {name!r}: {exclusions}")
        dropped = {column for group in exclusions for column in groups[group]}
        selected = [column for column in features if column not in dropped]
        if not selected:
            raise ValueError(f"Ablation {name!r} removes every feature.")
        resolved[name] = {"features": selected, "excluded_groups": exclusions}
    full = [name for name, definition in resolved.items() if not definition["excluded_groups"]]
    if len(full) != 1:
        raise ValueError("Exactly one ablation must be the full baseline with no exclusions.")
    # Always train the reference first, even when the JSON lists it elsewhere.
    return {full[0]: resolved[full[0]], **{name: value for name, value in resolved.items() if name != full[0]}}


def comparison_table(results, feature_sets):
    """Compare each retrained model against the same study's full baseline.

    A positive delta means removal improved validation performance. Count weekly
    Gini improvements only on weeks scored by both models. This is a screening
    table; it does not declare a final feature set or a significance threshold.
    """
    reference = results[next(iter(feature_sets))]
    reference_weeks = reference["weekly"].set_index("WEEK_NUM")["gini"]
    rows = []
    for name, result in results.items():
        weekly_delta = (result["weekly"].set_index("WEEK_NUM")["gini"] - reference_weeks).dropna()
        metrics = result["metrics"]
        rows.append({
            "configuration": name,
            "excluded_groups": ",".join(feature_sets[name]["excluded_groups"]) or "none",
            "feature_count": result["feature_count"],
            "best_iteration": result["best_iteration"],
            "validation_roc_auc": metrics["roc_auc"],
            "delta_roc_auc": metrics["roc_auc"] - reference["metrics"]["roc_auc"],
            "validation_gini_stability": metrics["gini_stability"],
            "delta_gini_stability": metrics["gini_stability"] - reference["metrics"]["gini_stability"],
            "weeks_better_than_baseline": int((weekly_delta > 0).sum()),
            "weeks_worse_than_baseline": int((weekly_delta < 0).sum()),
            "weeks_compared": len(weekly_delta),
            "run_id": result["run_id"],
            "model_uri": result["model_uri"],
        })
    return pd.DataFrame(rows)


def run_study(data, config, threads, run_name, *, method="group_ablation", nested=False, tags=None):
    """Train a shared-data study, optionally nested under a confirmation run.

    Return the study ID, individual results, and comparison table so a caller
    can compare combinations or aggregate paired results across time windows.
    The caller sets the tracking URI and experiment before invoking this function.
    """
    feature_sets = build_feature_sets(data.features, config)
    results = {}
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_name=run_name, nested=nested) as study:
        mlflow.set_tags({
            "stage": "feature_selection", "method": method,
            "result_status": "requires_temporal_confirmation",
            **(tags or {}),
        })
        mlflow.log_params({
            "configuration_count": len(feature_sets), "threads": threads,
            "primary_metric": "validation_gini_stability", **data.fingerprints,
        })
        mlflow.log_dict(config, "configuration.json")
        mlflow.log_dict(feature_sets, "feature_sets.json")
        mlflow.log_dict(data.splits, "split_weeks.json")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="provenance")
        print(f"Ablation study: {study.info.run_id}", flush=True)
        for name, definition in feature_sets.items():
            result = baseline.train_and_log(
                data, config, definition["features"], threads, name,
                stage="feature_selection", excluded_groups=definition["excluded_groups"], nested=True,
            )
            results[name] = result
            row = comparison_table(results, feature_sets).iloc[-1]
            for metric in ("delta_gini_stability", "delta_roc_auc", "weeks_better_than_baseline", "weeks_worse_than_baseline"):
                client.log_metric(result["run_id"], metric, float(row[metric]))
            mlflow.log_metric("completed_configurations", len(results))
            print(f"{name}: Gini stability {row['validation_gini_stability']:.6f}, delta {row['delta_gini_stability']:+.6f}", flush=True)

        comparison = comparison_table(results, feature_sets).sort_values("validation_gini_stability", ascending=False, kind="stable")
        mlflow.log_table(comparison, "ablation_comparison.json")
        best = comparison.iloc[0]
        mlflow.set_tag("best_observed_configuration", best["configuration"])
        mlflow.log_metric("best_observed_gini_stability", float(best["validation_gini_stability"]))
        mlflow.log_metric("best_observed_delta_gini_stability", float(best["delta_gini_stability"]))

    columns = ["configuration", "feature_count", "validation_roc_auc", "delta_roc_auc", "validation_gini_stability", "delta_gini_stability", "weeks_better_than_baseline", "weeks_compared"]
    print(comparison[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"MLflow study: {study.info.run_id}")
    print("Comparison table: ablation_comparison.json in the study's artifacts.")
    print("Confirm promising removals together and on additional chronological windows before changing eligibility.")
    return {"run_id": study.info.run_id, "results": results, "comparison": comparison}


def main():
    """Inspect a feature plan or execute it using the shared study runner."""
    parser = baseline.make_parser(__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Inspect feature sets without training or logging.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    config = baseline.load_config(args.config)
    features = baseline.eligible_features(args.processed_dir / "model_feature_eligibility.csv")
    feature_sets = build_feature_sets(features, config)
    if args.dry_run:
        for name, definition in feature_sets.items():
            print(f"{name}: {len(definition['features'])} features")
            for feature in definition["features"]:
                print(f"  {feature}")
        return
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    data = baseline.load_training_data(args.processed_dir, config, args.threads)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_study(data, config, args.threads, args.run_name or f"group-ablation-{timestamp}")


if __name__ == "__main__":
    main()
