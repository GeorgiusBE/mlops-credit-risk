"""Review joint removals, then validate explicitly chosen feature sets over time."""

import copy
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd

if __package__:
    from . import train_baseline as baseline
    from .select_features import build_feature_sets, comparison_table, run_study
else:
    import train_baseline as baseline
    from select_features import build_feature_sets, comparison_table, run_study


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "joint_confirmation.json"


def configure_joint_groups(config, groups):
    """Generate removal combinations only for groups explicitly chosen by a human.

    Keep the source configuration unchanged. Stable names use sorted group names,
    so changing the order of a CLI list does not change the experiment identity.
    Individual removals are reused from screening; only pairs and larger joint
    removals, plus a fresh full baseline, need fitting in this phase.
    """
    if not isinstance(groups, list) or not all(isinstance(name, str) for name in groups):
        raise ValueError("candidate_groups must be a list of group names.")
    if len(groups) < 2:
        raise ValueError("Choose at least two groups after group ablation; set CANDIDATE_GROUPS or candidate_groups in the configuration.")
    if len(groups) != len(set(groups)):
        raise ValueError("Candidate groups must not contain duplicates.")
    unknown = set(groups) - set(config["feature_groups"])
    if unknown:
        raise ValueError(f"Unknown candidate groups: {sorted(unknown)}")
    result = copy.deepcopy(config)
    result["candidate_groups"] = list(groups)
    result["ablations"] = {"all_eligible": []}
    for size in range(2, len(groups) + 1):
        for subset in combinations(sorted(groups), size):
            name = "without_" + "__and__".join(subset)
            result["ablations"][name] = list(subset)
    return result


def inherit_joint_plan(config, reviewed, requested_groups=None, *, inherit_screening=True):
    """Reuse the recorded group choices and labels from a reviewed study.

    This transfers an earlier decision, not a score-based shortlist. Preserve
    legacy configuration names so existing completed studies remain usable.
    Never inherit a temporal shortlist: the next handoff needs a new explicit
    human choice. Current model settings stay subject to the reuse checks.
    """
    recorded_groups = reviewed.get("candidate_groups", [])
    result = configure_joint_groups(config, recorded_groups)
    if requested_groups:
        configure_joint_groups(config, requested_groups)
        if set(requested_groups) != set(recorded_groups):
            raise ValueError("Candidate groups differ from the reviewed joint study; use its recorded groups or create a new joint study.")
    if "ablations" not in reviewed:
        raise ValueError("The reviewed study does not contain a joint-removal plan.")
    result["ablations"] = copy.deepcopy(reviewed["ablations"])
    if inherit_screening:
        result["screening_parent_run_id"] = reviewed["screening_parent_run_id"]
    return result


def window_config(config, window, ablations):
    """Create an expanding-window configuration with the original final test fixed.

    Future weeks between an early validation window and the final test remain
    unused. Neither their labels nor final test labels enter that window's fit.
    Validate through the same loader used by the standalone baseline.
    """
    result = copy.deepcopy(config)
    result["split"].update({key: window[key] for key in ("train_end", "validation_end")})
    result["ablations"] = copy.deepcopy(ablations)
    split = result["split"]
    if not split["train_start"] <= split["train_end"] < split["validation_end"] < split["test_start"] <= split["test_end"]:
        raise ValueError("Confirmation windows must be ordered and precede the fixed final test.")
    return result


def plan_confirmation(config, features):
    """Validate the joint-removal plan and all validation windows before training.

    Return all subset definitions, including the existing single-removal cases.
    Joint ablations must cover every pair/larger subset of the candidate groups.
    """
    groups = config["candidate_groups"]
    if len(groups) < 2 or len(groups) != len(set(groups)) or set(groups) - set(config["feature_groups"]):
        raise ValueError("candidate_groups must list distinct, known groups.")
    joint_sets = build_feature_sets(features, config)
    joint_exclusions = [frozenset(value["excluded_groups"]) for value in joint_sets.values()]
    expected_joint = {frozenset(), *(frozenset(combo) for size in range(2, len(groups) + 1) for combo in combinations(groups, size))}
    if len(joint_exclusions) != len(set(joint_exclusions)) or set(joint_exclusions) != expected_joint:
        raise ValueError("Joint ablations must include a baseline and every joint candidate-group removal exactly once.")
    reference_name = next(iter(joint_sets))
    complete = copy.deepcopy(config)
    complete["ablations"] = {
        reference_name: [],
        **{f"without_{group}": [group] for group in groups},
        **{name: value for name, value in config["ablations"].items() if value},
    }
    all_sets = build_feature_sets(features, complete)
    select_confirmation_sets(all_sets, config["confirmation"].get("selected_configurations", []), required=False)
    windows = config["confirmation"]["windows"]
    if len(windows) < 2 or len({window["name"] for window in windows}) != len(windows):
        raise ValueError("Use at least two uniquely named confirmation windows.")
    if "test_start" not in config["split"]:
        raise ValueError("Confirmation requires an explicit, fixed test_start.")
    ranges = []
    for window in windows:
        current = window_config(config, window, {reference_name: []})
        split = current["split"]
        if any(type(split[key]) is not int for key in ("train_end", "validation_end")):
            raise ValueError("Window boundaries must be integers.")
        weeks = set(range(split["train_end"] + 1, split["validation_end"] + 1))
        if any(weeks & previous for previous in ranges):
            raise ValueError("Confirmation validation windows must not overlap.")
        ranges.append(weeks)
    return all_sets


def select_confirmation_sets(feature_sets, requested, *, required=True):
    """Validate a human shortlist, retain its order, and add the baseline control.

    Never infer candidates from scores or a candidate count. An empty shortlist
    is valid while reviewing a plan, but cannot start temporal validation.
    """
    if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
        raise ValueError("selected_configurations must be a list of configuration names.")
    if not requested:
        if required:
            raise ValueError("Choose configurations after reviewing joint results; set CANDIDATES or confirmation.selected_configurations.")
        return {}
    if len(requested) != len(set(requested)):
        raise ValueError("Selected configurations must not contain duplicates.")
    unknown = set(requested) - set(feature_sets)
    if unknown:
        raise ValueError(f"Unknown selected configurations: {sorted(unknown)}")
    reference_name = next(iter(feature_sets))
    if reference_name in requested:
        raise ValueError("The full baseline is included automatically; select reduced configurations only.")
    return {name: feature_sets[name] for name in [reference_name, *requested]}


def read_json_artifact(client, run_id, name):
    """Read a small MLflow artifact through the server using temporary storage."""
    with TemporaryDirectory() as directory:
        return json.loads(Path(client.download_artifacts(run_id, name, directory)).read_text())


def resolve_joint_study_id(client, study_id):
    """Accept either a joint-review parent or its underlying fitted joint study.

    Review parents combine earlier single-removal results with a nested joint
    study. Their source manifest identifies the fitted study to reuse; the
    normal input, split, parameter, and membership checks still apply afterward.
    """
    if client.get_run(study_id).info.status != "FINISHED":
        raise ValueError("Review a completed joint study before temporal validation.")
    if "source_runs.json" in {artifact.path for artifact in client.list_artifacts(study_id)}:
        source = read_json_artifact(client, study_id, "source_runs.json")
        joint_id = source.get("joint_study_id")
        if not isinstance(joint_id, str) or not joint_id:
            raise ValueError("The review's source manifest does not identify a fitted joint study.")
        return joint_id
    return study_id


def validate_reuse(run, data, config, threads):
    """Reject prior results produced from different inputs or training settings."""
    expected = {
        **data.fingerprints, **config["model"], "n_jobs": threads,
        "early_stopping_rounds": config["early_stopping_rounds"],
        "train_cases": len(data.train), "validation_cases": len(data.validation),
    }
    if run.info.status != "FINISHED":
        raise ValueError(f"Cannot reuse unfinished run {run.info.run_id}.")
    mismatches = [key for key, value in expected.items() if run.data.params.get(key) != str(value)]
    if mismatches:
        raise ValueError(f"Prior run {run.info.run_id} has incompatible settings: {mismatches}")


def load_study_results(client, study_id, data, config, threads, feature_sets):
    """Reuse eligible completed models after checking provenance and exact splits.

    Read full-precision metrics from the tracking backend rather than the rounded
    comparison artifact. Reconstruct weekly results from their metric histories.
    Reuse supplies earlier single removals or resumes a completed joint study;
    it never substitutes earlier-window models for new temporal fits.
    """
    if client.get_run(study_id).info.status != "FINISHED":
        raise ValueError(f"Study {study_id} must be complete before reuse.")
    saved = read_json_artifact(client, study_id, "ablation_comparison.json")
    table = pd.DataFrame(saved["data"], columns=saved["columns"])
    results = {}
    for row in table.itertuples(index=False):
        if row.configuration not in feature_sets:
            continue
        run = client.get_run(row.run_id)
        if run.data.tags.get("mlflow.parentRunId") != study_id:
            raise ValueError("The saved comparison references a run outside its source study.")
        validate_reuse(run, data, config, threads)
        if read_json_artifact(client, row.run_id, "split_weeks.json") != data.splits:
            raise ValueError("Prior results use different week splits.")
        membership = read_json_artifact(client, row.run_id, "features.json")
        if membership["features"] != feature_sets[row.configuration]["features"]:
            raise ValueError("Prior results use different feature membership or order.")
        runtime = read_json_artifact(client, row.run_id, "runtime_versions.json")
        if any(runtime[name] != version(name) for name in ("lightgbm", "numpy", "pandas", "scikit-learn")):
            raise ValueError("Runtime versions changed; rerun the screening study before reuse.")
        histories = {
            name: {point.step: point.value for point in client.get_metric_history(row.run_id, f"validation_weekly_{name}")}
            for name in ("cases", "defaults", "roc_auc", "gini")
        }
        weekly = pd.DataFrame([
            {"WEEK_NUM": week, **{name: values.get(week, np.nan) for name, values in histories.items()}}
            for week in data.splits["validation"]
        ])
        if weekly[["cases", "defaults"]].isna().any().any():
            raise ValueError("Prior weekly metric histories are incomplete.")
        results[row.configuration] = {
            "run_id": row.run_id, "model_uri": row.model_uri,
            "feature_count": int(run.data.params["feature_count"]),
            "best_iteration": int(run.data.metrics["best_iteration"]), "weekly": weekly,
            "metrics": {key.removeprefix("validation_"): value for key, value in run.data.metrics.items()
                        if key.startswith("validation_") and not key.startswith("validation_weekly_")},
        }
    return results


def summarize_windows(comparisons, reference_name):
    """Aggregate paired candidate-minus-baseline changes within each window.

    Use each window's own baseline, never the latest window's absolute score.
    Require complete coverage and a consistent feature count for each candidate.
    """
    if len(comparisons) < 2:
        raise ValueError("At least two confirmation windows are required.")
    paired = []
    candidates = None
    for name, frame in comparisons.items():
        if frame["configuration"].duplicated().any() or reference_name not in set(frame["configuration"]):
            raise ValueError("Each window needs distinct configurations and its own baseline.")
        current = set(frame["configuration"])
        if candidates is not None and current != candidates:
            raise ValueError("Every candidate must be evaluated in every confirmation window.")
        candidates = current
        reference = frame.set_index("configuration").loc[reference_name]
        for row in frame.to_dict("records"):
            paired.append({
                **row, "window": name,
                "delta_gini_stability": row["validation_gini_stability"] - reference["validation_gini_stability"],
                "delta_roc_auc": row["validation_roc_auc"] - reference["validation_roc_auc"],
            })
    details = pd.DataFrame(paired)
    rows = []
    for name, group in details.groupby("configuration", sort=False):
        if group["feature_count"].nunique() != 1:
            raise ValueError("Feature counts must remain fixed across confirmation windows.")
        rows.append({
            "configuration": name, "feature_count": int(group["feature_count"].iloc[0]),
            "mean_delta_gini_stability": float(group["delta_gini_stability"].mean()),
            "worst_delta_gini_stability": float(group["delta_gini_stability"].min()),
            "std_delta_gini_stability": float(group["delta_gini_stability"].std(ddof=0)),
            "mean_delta_roc_auc": float(group["delta_roc_auc"].mean()),
            "worst_delta_roc_auc": float(group["delta_roc_auc"].min()),
            "windows_improved": int(group["delta_gini_stability"].gt(0).sum()),
            "windows_evaluated": len(group),
        })
    summary = pd.DataFrame(rows).sort_values("mean_delta_gini_stability", ascending=False, kind="stable")
    return summary, details


def log_joint_plot(combinations_table):
    """Log the comparison plot used for human review before temporal fits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with TemporaryDirectory() as directory:
        table = combinations_table.sort_values("delta_gini_stability")
        fig, axis = plt.subplots(figsize=(11, 6), layout="constrained")
        colors = ["#27866b" if value >= 0 else "#c65b57" for value in table["delta_gini_stability"]]
        axis.barh(table["configuration"].str.replace("_", " "), table["delta_gini_stability"], color=colors)
        axis.axvline(0, color="#444444", linewidth=1)
        axis.set(xlabel="Change in validation Gini stability versus full baseline", title="Individual and joint removals — weeks 64–77")
        path = Path(directory) / "joint_removal_deltas.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        mlflow.log_artifact(str(path))


def log_plots(combinations_table, details, window_results, reference_name):
    """Log the comparison evidence without making a final feature decision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_joint_plot(combinations_table)
    with TemporaryDirectory() as directory:
        reduced = details.loc[details["configuration"].ne(reference_name)]
        pivot = reduced.pivot(index="window", columns="configuration", values="delta_gini_stability")
        axis = pivot.rename(columns=lambda name: name.replace("_", " ")).plot.bar(figsize=(11, 6), rot=0)
        axis.axhline(0, color="#444444", linewidth=1)
        axis.set(xlabel="Chronological validation window", ylabel="Candidate minus baseline Gini stability", title="Consistency of the candidate removals")
        axis.figure.tight_layout()
        path = Path(directory) / "confirmation_deltas.png"
        axis.figure.savefig(path, dpi=160)
        plt.close(axis.figure)
        mlflow.log_artifact(str(path))

        fig, axes = plt.subplots(1, len(window_results), figsize=(16, 4), squeeze=False, layout="constrained")
        for axis, (name, results) in zip(axes[0], window_results.items()):
            for configuration, result in results.items():
                weekly = result["weekly"]
                axis.plot(weekly["WEEK_NUM"], weekly["gini"], marker=".", label=configuration.replace("_", " "))
            axis.set(title=name.replace("_", " "), xlabel="WEEK_NUM", ylabel="Weekly validation Gini")
            axis.grid(alpha=0.2)
        axes[0, -1].legend(fontsize=7)
        path = Path(directory) / "confirmation_weekly_gini.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        mlflow.log_artifact(str(path))


def main():
    """Run joint review or temporal validation with a human-provided shortlist.

    Joint review stops after logging comparisons. Temporal validation requires
    a completed joint/review study and explicit configuration names. Metrics
    and plots support human judgement; no final feature set is selected here.
    """
    parser = baseline.make_parser(__doc__)
    parser.set_defaults(config=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("joint", "temporal"), default="temporal", help="Joint comparison for review, or temporal validation of your chosen sets.")
    parser.add_argument("--screening-study-id", help="Override the original group-ablation study ID stored in the configuration.")
    parser.add_argument("--joint-study-id", help="Completed joint-review or fitted joint-study ID; required for temporal validation.")
    parser.add_argument("--candidate-groups", nargs="+", help="Groups you choose after group ablation; generates their joint combinations.")
    parser.add_argument("--candidates", nargs="+", help="Configuration names you choose for temporal validation; overrides the config list.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without training or logging runs; may read MLflow metadata when a joint study ID is supplied.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    config = baseline.load_config(args.config)
    if args.screening_study_id:
        config["screening_parent_run_id"] = args.screening_study_id
    if args.candidates is not None:
        config["confirmation"]["selected_configurations"] = list(args.candidates)
    features = baseline.eligible_features(args.processed_dir / "model_feature_eligibility.csv")
    requested = config["confirmation"].get("selected_configurations", [])
    groups = args.candidate_groups if args.candidate_groups is not None else config.get("candidate_groups", [])
    client = None
    try:
        if args.stage == "temporal" and not args.dry_run:
            if not requested:
                raise ValueError("Choose configurations after reviewing joint results; set CANDIDATES or confirmation.selected_configurations.")
            if not args.joint_study_id:
                raise ValueError("Temporal validation requires JOINT_STUDY_ID. Run make joint, review its results, and supply its study ID with your chosen CANDIDATES.")
        if args.stage == "temporal" and args.joint_study_id:
            mlflow.set_tracking_uri(args.tracking_uri)
            client = mlflow.MlflowClient()
            reviewed = read_json_artifact(client, args.joint_study_id, "configuration.json")
            config = inherit_joint_plan(config, reviewed, groups, inherit_screening=not args.screening_study_id)
        elif not groups and args.dry_run:
            print("Available groups (choose after reviewing group ablation):")
            for name, columns in config["feature_groups"].items():
                print(f"  {name}: {len(columns)} features")
            print("No candidate groups have been chosen; there is no default shortlist.")
            if args.stage == "joint":
                print('Next: make joint-plan CANDIDATE_GROUPS="your_chosen_groups"')
            else:
                print("Supply a reviewed JOINT_STUDY_ID to load its recorded groups, or CANDIDATE_GROUPS for an offline preview.")
            return
        else:
            config = configure_joint_groups(config, groups)
        all_sets = plan_confirmation(config, features)
        reference_name = next(iter(all_sets))
        selected_sets = select_confirmation_sets(all_sets, requested, required=args.stage == "temporal" and not args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print("Joint phase:")
        for name, definition in build_feature_sets(features, config).items():
            print(f"  {name}: {len(definition['features'])} features")
        print(f"Screening study: {config['screening_parent_run_id']}")
        print(f"Human-selected groups: {', '.join(config['candidate_groups'])}")
        print(f"Reuse {len(config['candidate_groups'])} compatible individual-removal cases from screening.")
        if args.stage == "joint":
            print("Stop after the joint comparison for human review; no temporal fits will run.")
        else:
            print(f"Reviewed joint study: {args.joint_study_id or '(not supplied)'}")
            print("Human-selected configurations: " + (", ".join(requested) if requested else "(none — awaiting your decision)"))
            print("Available reduced configurations:")
            for name, definition in all_sets.items():
                if name != reference_name:
                    print(f"  {name}: {len(definition['features'])} features")
        print("Temporal windows (full baseline is always the control):")
        for window in config["confirmation"]["windows"]:
            print(f"  {window['name']}: train through {window['train_end']}, validate through {window['validation_end']}")
        print(f"Final test stays at weeks {config['split']['test_start']}–{config['split']['test_end']}.")
        return

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    if client is None:
        client = mlflow.MlflowClient()
    data = baseline.load_training_data(args.processed_dir, config, args.threads)
    prior = load_study_results(client, config["screening_parent_run_id"], data, config, args.threads, all_sets)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with mlflow.start_run(run_name=args.run_name or f"{args.stage}-review-{timestamp}") as study:
        mlflow.set_tags({
            "stage": "feature_selection", "method": "joint_review" if args.stage == "joint" else "human_selected_temporal_validation",
            "candidate_selection": "human",
        })
        mlflow.log_params({
            "screening_parent_run_id": config["screening_parent_run_id"],
            "final_test_week_start": config["split"]["test_start"], "final_test_week_end": config["split"]["test_end"],
            "candidate_selection": "human", **data.fingerprints,
        })
        mlflow.log_dict(config, "configuration.json")
        mlflow.log_dict(all_sets, "available_feature_sets.json")
        mlflow.log_dict({
            "selection_method": "human", "candidate_groups": config["candidate_groups"],
            "selection_source": "reviewed_study" if args.stage == "temporal" else ("command_line" if args.candidate_groups is not None else "configuration"),
            "screening_parent_run_id": config["screening_parent_run_id"],
        }, "joint_group_selection.json")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="provenance")
        if args.joint_study_id:
            joint_id = resolve_joint_study_id(client, args.joint_study_id)
            joint_results = load_study_results(client, joint_id, data, config, args.threads, all_sets)
        else:
            joint = run_study(data, config, args.threads, "joint_removals", method="joint_ablation", nested=True)
            joint_results, joint_id = joint["results"], joint["run_id"]
        if not set(config["ablations"]) <= set(joint_results):
            raise ValueError("The joint study does not contain the required joint-removal runs.")
        for metric in ("roc_auc", "gini_stability", "average_precision", "log_loss"):
            if not np.isclose(prior[reference_name]["metrics"][metric], joint_results[reference_name]["metrics"][metric], rtol=0, atol=1e-12):
                raise ValueError("Fresh baseline does not reproduce screening; investigate before combining studies.")
        combined = {name: joint_results.get(name, prior.get(name)) for name in all_sets}
        if any(value is None for value in combined.values()):
            raise ValueError("Screening and joint studies do not cover every removal pattern.")
        combination_table = comparison_table(combined, all_sets).sort_values(
            ["validation_gini_stability", "validation_roc_auc", "feature_count"], ascending=[False, False, True], kind="stable",
        )
        mlflow.log_table(combination_table, "combination_comparison.json")
        mlflow.log_dict({"joint_study_id": joint_id, "runs": {name: result["run_id"] for name, result in combined.items()}}, "source_runs.json")
        if args.stage == "joint":
            log_joint_plot(combination_table)
            mlflow.set_tag("result_status", "awaiting_human_shortlist")
            print(combination_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
            print(f"Joint review study: {study.info.run_id}")
            print("Review combination_comparison.json and joint_removal_deltas.png before choosing candidates.")
            print(f'Next: make confirm JOINT_STUDY_ID={study.info.run_id} CANDIDATES="your_chosen_configuration_names"')
            return
        mlflow.log_dict(selected_sets, "confirmation_candidates.json")
        mlflow.log_dict({
            "selection_method": "human", "selected_configurations": requested,
            "selection_source": "command_line" if args.candidates is not None else "configuration",
            "reviewed_study_id": args.joint_study_id, "baseline_control": reference_name,
        }, "confirmation_selection.json")
        mlflow.log_param("reviewed_study_id", args.joint_study_id)
        fingerprints = data.fingerprints
        screening_splits = data.splits
        del data
        comparisons, window_results = {}, {}
        for window in config["confirmation"]["windows"]:
            current = window_config(config, window, {name: definition["excluded_groups"] for name, definition in selected_sets.items()})
            if current["split"] == config["split"]:
                # These exact models were already fitted and their provenance
                # checked above. Reuse this window instead of training copies.
                reused = {name: combined[name] for name in selected_sets}
                table = comparison_table(reused, selected_sets)
                with mlflow.start_run(run_name=window["name"], nested=True):
                    mlflow.set_tags({
                        "stage": "feature_selection", "method": "temporal_window",
                        "validation_window": window["name"], "reused_existing_models": "true",
                    })
                    mlflow.log_params({"configuration_count": len(reused), **fingerprints})
                    mlflow.log_dict(current, "configuration.json")
                    mlflow.log_dict(selected_sets, "feature_sets.json")
                    mlflow.log_dict(screening_splits, "split_weeks.json")
                    mlflow.log_dict({name: result["run_id"] for name, result in reused.items()}, "source_runs.json")
                    mlflow.log_table(table, "ablation_comparison.json")
                comparisons[window["name"]], window_results[window["name"]] = table, reused
                print(f"Reused compatible fitted models for {window['name']}.", flush=True)
                continue
            window_data = baseline.load_training_data(args.processed_dir, current, args.threads)
            if window_data.fingerprints != fingerprints:
                raise ValueError("Input data changed during confirmation.")
            result = run_study(
                window_data, current, args.threads, window["name"], method="temporal_window", nested=True,
                tags={"validation_window": window["name"]},
            )
            comparisons[window["name"]] = result["comparison"]
            window_results[window["name"]] = result["results"]
            del window_data

        summary, details = summarize_windows(comparisons, reference_name)
        mlflow.log_table(details, "confirmation_by_window.json")
        mlflow.log_table(summary, "confirmation_summary.json")
        for row in summary.to_dict("records"):
            with mlflow.start_run(run_name=f"summary-{row['configuration']}", nested=True):
                mlflow.set_tags({"stage": "feature_selection_summary", "configuration": row["configuration"]})
                mlflow.log_param("feature_count", row["feature_count"])
                mlflow.log_metrics({key: float(value) for key, value in row.items() if key not in ("configuration", "feature_count")})
        mlflow.set_tag("result_status", "awaiting_human_feature_decision")
        log_plots(combination_table, details, window_results, reference_name)
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("Review the temporal summaries and plots to decide the final feature set.")
    print(f"Confirmation study: {study.info.run_id}")


if __name__ == "__main__":
    main()
