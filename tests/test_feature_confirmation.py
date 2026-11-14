"""Check paired temporal comparisons, reuse constraints, and final-test isolation."""

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_selection import train_baseline as baseline
from feature_selection import confirm_features as confirmation
from feature_selection.confirm_features import (
    DEFAULT_CONFIG, configure_joint_groups, inherit_joint_plan, plan_confirmation, resolve_joint_study_id,
    select_confirmation_sets, summarize_windows, validate_reuse, window_config,
)


TRIPLE_REMOVAL = "without_affordability_ratios__and__debit_card_history__and__deposit_history"


@pytest.fixture
def raw_config():
    """Load the plan without reading outcome data or contacting MLflow."""
    return baseline.load_config(DEFAULT_CONFIG)


@pytest.fixture
def joint_groups():
    return ["deposit_history", "affordability_ratios", "debit_card_history"]


@pytest.fixture
def joint_config(raw_config, joint_groups):
    return configure_joint_groups(raw_config, joint_groups)


@pytest.fixture
def eligible_features(joint_config):
    return [name for columns in joint_config["feature_groups"].values() for name in columns]


@pytest.mark.parametrize("groups", [
    pytest.param([], id="empty"),
    pytest.param(["deposit_history"], id="single-group"),
    pytest.param(["deposit_history", "deposit_history"], id="duplicate"),
    pytest.param(["deposit_history", "unknown"], id="unknown"),
    pytest.param("deposit_history", id="string-instead-of-list"),
    pytest.param(["deposit_history", "without_deposit_history"], id="duplicate-alias"),
    pytest.param(["without_deposit_history", "without_unknown"], id="unknown-alias"),
    pytest.param(
        ["without_debit_card_history__and__deposit_history", "without_affordability_ratios"],
        id="combined-removal",
    ),
])
def test_joint_groups_require_explicit_valid_choices(raw_config, groups):
    """No default shortlist may silently advance groups from screening."""
    assert raw_config["candidate_groups"] == []
    assert "ablations" not in raw_config
    with pytest.raises(ValueError):
        configure_joint_groups(raw_config, groups)


def test_joint_cli_requires_explicit_groups(raw_config, eligible_features):
    """Missing joint choices must fail before loading data or contacting MLflow."""
    with ExitStack() as stack:
        stack.enter_context(patch("sys.argv", ["confirm_features.py", "--stage", "joint"]))
        stack.enter_context(patch.object(baseline, "load_config", return_value=raw_config))
        stack.enter_context(patch.object(baseline, "eligible_features", return_value=eligible_features))
        load = stack.enter_context(patch.object(baseline, "load_training_data"))
        tracking = stack.enter_context(patch.object(confirmation.mlflow, "set_tracking_uri"))
        error = stack.enter_context(redirect_stderr(io.StringIO()))
        with pytest.raises(SystemExit) as exit_status:
            confirmation.main()
        assert exit_status.value.code == 2
        assert "Choose at least two groups" in error.getvalue()
        load.assert_not_called()
        tracking.assert_not_called()


@pytest.mark.parametrize("prefix_style", ["prefixed", "mixed"])
def test_prefixed_joint_groups_preserve_membership_and_existing_studies(
    raw_config, joint_groups, joint_config, eligible_features, prefix_style,
):
    """Accept copied single-removal run names in CLI/config and reuse checks."""
    prefixed = [f"without_{group}" for group in joint_groups]
    groups = prefixed if prefix_style == "prefixed" else [joint_groups[0], *prefixed[1:]]
    configured = configure_joint_groups(raw_config, groups)
    assert configured == joint_config
    assert plan_confirmation(configured, eligible_features) == plan_confirmation(joint_config, eligible_features)
    inherited = inherit_joint_plan(raw_config, joint_config, groups)
    assert inherited["candidate_groups"] == joint_groups
    assert prefixed == [f"without_{group}" for group in joint_groups]
    config = copy.deepcopy(raw_config)
    config["candidate_groups"] = prefixed
    assert configure_joint_groups(config, config["candidate_groups"]) == joint_config
    assert config["candidate_groups"] == prefixed
    reviewed = copy.deepcopy(joint_config)
    reviewed["candidate_groups"] = prefixed
    assert inherit_joint_plan(raw_config, reviewed, joint_groups)["candidate_groups"] == joint_groups


def test_joint_combinations_follow_arbitrary_human_group_choices(raw_config, joint_groups, eligible_features):
    """Generate only chosen subsets, with stable labels and no three-group limit."""
    groups = ["loan_amounts", "applicant_information"]
    config = configure_joint_groups(raw_config, groups)
    sets = plan_confirmation(config, eligible_features)
    assert len(config["ablations"]) == 2  # Fresh baseline and one pair.
    assert len(sets) == 4  # Plus the reused single removals.
    pair = sets["without_applicant_information__and__loan_amounts"]
    assert set(pair["excluded_groups"]) == set(groups)
    assert len(pair["features"]) == 14
    reversed_config = configure_joint_groups(raw_config, list(reversed(groups)))
    assert config["ablations"] == reversed_config["ablations"]
    four = configure_joint_groups(raw_config, [*joint_groups, "loan_amounts"])
    assert len(four["ablations"]) == 12
    assert len(plan_confirmation(four, eligible_features)) == 16
    assert raw_config["candidate_groups"] == []
    assert "ablations" not in raw_config


def test_reviewed_plan_preserves_choices_and_legacy_labels_without_shortlisting(raw_config, joint_groups, joint_config, eligible_features):
    """Reuse previous group decisions and names, but require a fresh shortlist."""
    reviewed = copy.deepcopy(joint_config)
    reviewed["ablations"]["legacy_triple_name"] = reviewed["ablations"].pop(TRIPLE_REMOVAL)
    reviewed["confirmation"]["selected_configurations"] = ["legacy_triple_name"]
    reviewed["screening_parent_run_id"] = "reviewed-screening"
    reviewed["model"]["learning_rate"] = 0.1
    inherited = inherit_joint_plan(raw_config, reviewed)
    assert inherited["candidate_groups"] == joint_groups
    assert "legacy_triple_name" in plan_confirmation(inherited, eligible_features)
    assert inherited["confirmation"]["selected_configurations"] == []
    assert inherited["model"] == raw_config["model"]
    assert inherited["screening_parent_run_id"] == "reviewed-screening"
    assert inherit_joint_plan(raw_config, reviewed, inherit_screening=False)["screening_parent_run_id"] == raw_config["screening_parent_run_id"]
    inherit_joint_plan(raw_config, reviewed, list(reversed(joint_groups)))
    with pytest.raises(ValueError, match="differ from the reviewed"):
        inherit_joint_plan(raw_config, reviewed, ["loan_amounts", "deposit_history"])


def test_all_eight_patterns_and_final_test_are_explicit(joint_config, eligible_features):
    """Account for the baseline, three singles, three pairs, and triple drop."""
    sets = plan_confirmation(joint_config, eligible_features)
    assert len(sets) == 8
    assert len(sets["without_debit_card_history__and__deposit_history"]["features"]) == 14
    assert len(sets[TRIPLE_REMOVAL]["features"]) == 12
    for window in joint_config["confirmation"]["windows"]:
        current = window_config(joint_config, window, {"all_eligible": []})
        assert current["split"]["test_start"] == 78
        assert current["split"]["test_end"] == 91
        assert current["split"]["validation_end"] < 78
    incomplete = copy.deepcopy(joint_config)
    del incomplete["ablations"][TRIPLE_REMOVAL]
    with pytest.raises(ValueError, match="every joint"):
        plan_confirmation(incomplete, eligible_features)


def test_reject_overlapping_windows_and_final_test_validation(joint_config, eligible_features):
    """A confirmation window must not consume another validation window or test."""
    bad = copy.deepcopy(joint_config)
    bad["confirmation"]["windows"][1]["train_end"] = 40
    with pytest.raises(ValueError, match="must not overlap"):
        plan_confirmation(bad, eligible_features)
    with pytest.raises(ValueError, match="precede the fixed final test"):
        window_config(joint_config, {"train_end": 63, "validation_end": 78}, {"all_eligible": []})


def test_loader_skips_future_gap_and_retains_original_test(joint_config, tmp_path):
    """Unused future labels and final test labels must stay out of loaded frames."""
    config = copy.deepcopy(joint_config)
    config["split"] = {"train_start": 0, "train_end": 0, "validation_end": 1, "test_start": 3, "test_end": 3}
    source = pd.DataFrame({
        "WEEK_NUM": [0, 0, 1, 1, 2, 2, 3, 3],
        "target": [0, 1, 0, 1, 777, 777, 999, 999],
        "amount": list(range(8)),
    })
    source.to_parquet(tmp_path / "train_base_engineered.parquet", index=False)
    pd.DataFrame({"column_name": ["amount"], "role": ["eligible"]}).to_csv(tmp_path / "model_feature_eligibility.csv", index=False)
    data = baseline.load_training_data(tmp_path, config, 1)
    assert data.splits == {"train": [0], "validation": [1], "held_out_test": [3], "unused_future": [2]}
    assert data.train["amount"].tolist() == [0, 1]
    assert data.validation["amount"].tolist() == [2, 3]


def test_paired_deltas_use_each_windows_baseline():
    """Different absolute baseline scores must not distort paired changes."""
    first = pd.DataFrame([
        {"configuration": "full", "feature_count": 20, "validation_gini_stability": 0.50, "validation_roc_auc": 0.75},
        {"configuration": "reduced", "feature_count": 14, "validation_gini_stability": 0.52, "validation_roc_auc": 0.76},
    ])
    second = pd.DataFrame([
        {"configuration": "full", "feature_count": 20, "validation_gini_stability": 0.20, "validation_roc_auc": 0.60},
        {"configuration": "reduced", "feature_count": 14, "validation_gini_stability": 0.19, "validation_roc_auc": 0.58},
    ])
    summary, _ = summarize_windows({"early": first, "late": second}, "full")
    rows = summary.set_index("configuration")
    assert rows.loc["reduced", "mean_delta_gini_stability"] == pytest.approx(0.005, rel=0, abs=5e-8)
    assert rows.loc["reduced", "worst_delta_gini_stability"] == pytest.approx(-0.01, rel=0, abs=5e-8)
    assert rows.loc["reduced", "windows_improved"] == 1
    assert rows.loc["full", "mean_delta_gini_stability"] == 0
    with pytest.raises(ValueError, match="every confirmation window"):
        summarize_windows({"early": first, "late": second.iloc[:1]}, "full")


def test_manual_shortlist_has_no_automatic_count_or_ranking(joint_config, eligible_features):
    """Keep any valid human choices and their order, even beyond the old top two."""
    joint_config["confirmation"]["candidate_count"] = 1  # Legacy settings cannot select anything.
    sets = plan_confirmation(joint_config, eligible_features)
    names = [TRIPLE_REMOVAL, "without_affordability_ratios", "without_deposit_history"]
    selected = select_confirmation_sets(sets, names)
    assert list(selected) == ["all_eligible", *names]
    assert select_confirmation_sets(sets, [], required=False) == {}
    with pytest.raises(ValueError, match="Choose configurations"):
        select_confirmation_sets(sets, [])
    for invalid in (["not_a_configuration"], [names[0], names[0]], ["all_eligible"]):
        with pytest.raises(ValueError):
            select_confirmation_sets(sets, invalid)


def test_shortlist_accepts_optional_prefix_and_rejects_duplicate_aliases(joint_config, eligible_features):
    """Resolve short names to existing model identities before fitting or reuse."""
    names = ["debit_card_history__and__deposit_history", "deposit_history"]
    joint_config["confirmation"]["selected_configurations"] = names
    sets = plan_confirmation(joint_config, eligible_features)
    expected = ["all_eligible", *(f"without_{name}" for name in names)]
    selected = select_confirmation_sets(sets, names)
    assert list(selected) == expected
    assert selected == select_confirmation_sets(sets, expected[1:])
    assert selected == select_confirmation_sets(sets, [names[0], expected[2]])
    assert names == ["debit_card_history__and__deposit_history", "deposit_history"]
    with pytest.raises(ValueError, match="duplicates"):
        select_confirmation_sets(sets, ["deposit_history", "without_deposit_history"])
    with pytest.raises(ValueError, match="Available reduced configurations"):
        select_confirmation_sets(sets, ["unknown_group"])
    legacy = {"full": {}, "without_legacy_label": {}, "custom_label": {}}
    assert list(select_confirmation_sets(legacy, ["legacy_label", "custom_label"])) == list(legacy)


@pytest.mark.parametrize("arguments,message", [
    pytest.param(["--joint-study-id", "reviewed-study"], "Choose configurations", id="missing-shortlist"),
    pytest.param(["--candidates", "without_deposit_history"], "requires JOINT_STUDY_ID", id="missing-review"),
])
def test_temporal_cli_requires_human_choice_and_review_before_data_loading(
    joint_config, eligible_features, arguments, message,
):
    """Missing choices or review ID must fail before MLflow calls or training."""
    with ExitStack() as stack:
        stack.enter_context(patch("sys.argv", ["confirm_features.py", *arguments]))
        stack.enter_context(patch.object(baseline, "load_config", return_value=copy.deepcopy(joint_config)))
        stack.enter_context(patch.object(baseline, "eligible_features", return_value=eligible_features))
        load = stack.enter_context(patch.object(baseline, "load_training_data"))
        tracking = stack.enter_context(patch.object(confirmation.mlflow, "set_tracking_uri"))
        error = stack.enter_context(redirect_stderr(io.StringIO()))
        with pytest.raises(SystemExit) as exit_status:
            confirmation.main()
        assert exit_status.value.code == 2
        assert message in error.getvalue()
        load.assert_not_called()
        tracking.assert_not_called()


@pytest.mark.parametrize("stage", ["joint", "temporal"])
def test_joint_review_stops_and_temporal_uses_only_explicit_choices(
    raw_config, joint_groups, joint_config, eligible_features, stage, tmp_path,
):
    """Exercise both stages with logging/fits mocked, including the review pause."""
    sets = plan_confirmation(joint_config, eligible_features)
    results = {
        name: {
            "run_id": f"run-{name}", "model_uri": f"models:/{name}",
            "feature_count": len(definition["features"]), "best_iteration": 10,
            "metrics": {"roc_auc": 0.65, "gini_stability": 0.3, "average_precision": 0.2, "log_loss": 0.5},
            "weekly": pd.DataFrame({"WEEK_NUM": [64, 65], "gini": [0.2, 0.3]}),
        }
        for name, definition in sets.items()
    }
    data = SimpleNamespace(features=eligible_features, fingerprints={"dataset_sha256": "same-data"}, splits={"train": list(range(64)), "validation": list(range(64, 78)), "held_out_test": list(range(78, 92))})
    chosen = [TRIPLE_REMOVAL, "without_affordability_ratios"]
    with ExitStack() as stack:
        tracking_path = tmp_path / "test.db"
        previous_uri = confirmation.mlflow.get_tracking_uri()
        confirmation.mlflow.set_tracking_uri(f"sqlite:///{tracking_path}")
        stack.callback(confirmation.mlflow.set_tracking_uri, previous_uri)
        arguments = ["confirm_features.py", "--stage", stage]
        if stage == "temporal":
            arguments += ["--joint-study-id", "reviewed-study", "--candidates", *(name.removeprefix("without_") for name in chosen)]
        else:
            arguments += ["--candidate-groups", *(f"without_{group}" for group in joint_groups)]
        stack.enter_context(patch("sys.argv", arguments))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(baseline, "load_config", return_value=copy.deepcopy(raw_config)))
        stack.enter_context(patch.object(baseline, "eligible_features", return_value=eligible_features))
        loader = stack.enter_context(patch.object(baseline, "load_training_data", return_value=data))
        stack.enter_context(patch.object(confirmation, "load_study_results", return_value=results))
        stack.enter_context(patch.object(confirmation, "read_json_artifact", return_value=copy.deepcopy(joint_config)))
        stack.enter_context(patch.object(confirmation, "resolve_joint_study_id", return_value="fitted-joint"))
        stack.enter_context(patch.object(confirmation, "log_joint_plot"))
        stack.enter_context(patch.object(confirmation, "log_plots"))
        for name in ("set_tracking_uri", "set_experiment", "MlflowClient", "set_tags", "log_params", "log_param", "log_metrics", "log_dict", "log_artifact", "log_table", "set_tag"):
            stack.enter_context(patch.object(confirmation.mlflow, name))
        start = stack.enter_context(patch.object(confirmation.mlflow, "start_run"))
        start.return_value.__enter__.return_value.info.run_id = "review-parent"
        def fit_stub(_data, current, _threads, _name, **_kwargs):
            """Return only configurations requested by this stage."""
            selected = {name: results[name] for name in current["ablations"]}
            return {"run_id": "fitted-joint", "results": selected, "comparison": confirmation.comparison_table(selected, {name: sets[name] for name in selected})}
        fits = stack.enter_context(patch.object(confirmation, "run_study", side_effect=fit_stub))
        confirmation.main()
        recorded = {call.args[1]: call.args[0] for call in confirmation.mlflow.log_dict.call_args_list}
        assert recorded["available_feature_sets.json"] == sets
        assert recorded["joint_group_selection.json"]["candidate_groups"] == joint_groups
        assert recorded["joint_group_selection.json"]["selection_method"] == "human"
        assert recorded["joint_group_selection.json"]["selection_source"] == ("command_line" if stage == "joint" else "reviewed_study")
        if stage == "joint":
            loader.assert_called_once()
            fits.assert_called_once()
            confirmation.log_joint_plot.assert_called_once()
            confirmation.log_plots.assert_not_called()
            assert "confirmation_candidates.json" not in recorded
            confirmation.mlflow.set_tag.assert_called_with("result_status", "awaiting_human_shortlist")
        else:
            assert fits.call_count == 2  # Earlier windows; latest reuses models.
            for call in fits.call_args_list:
                assert list(call.args[1]["ablations"]) == ["all_eligible", *chosen]
            assert recorded["confirmation_selection.json"]["selected_configurations"] == chosen
            assert recorded["confirmation_selection.json"]["requested_configurations"] == [name.removeprefix("without_") for name in chosen]
            assert recorded["configuration.json"]["confirmation"]["selected_configurations"] == chosen
            assert recorded["confirmation_selection.json"]["selection_method"] == "human"
            assert "recommendation.json" not in recorded
            confirmation.mlflow.set_tag.assert_called_with("result_status", "awaiting_human_feature_decision")
        assert not tracking_path.exists(), "Mocked workflow must not create real tracking records."


def test_resolve_completed_review_to_fitted_joint_source():
    """A review's manifest must resolve to a checked fitted joint study."""
    client = MagicMock()
    client.get_run.return_value.info.status = "FINISHED"
    client.list_artifacts.return_value = [SimpleNamespace(path="source_runs.json")]
    with patch.object(confirmation, "read_json_artifact", return_value={"joint_study_id": "fitted-study"}):
        assert resolve_joint_study_id(client, "reviewed-study") == "fitted-study"
    client.list_artifacts.return_value = []
    assert resolve_joint_study_id(client, "fitted-study") == "fitted-study"


def test_refuse_reuse_after_dataset_or_parameter_change(joint_config):
    """Old screening results require matching data, settings, and completion."""
    data = SimpleNamespace(fingerprints={"dataset_sha256": "data", "feature_inventory_sha256": "inventory"}, train=[1, 2], validation=[1, 2])
    params = {key: str(value) for key, value in {
        **joint_config["model"], **data.fingerprints, "n_jobs": 2,
        "early_stopping_rounds": 40, "train_cases": 2, "validation_cases": 2,
    }.items()}
    run = SimpleNamespace(info=SimpleNamespace(status="FINISHED", run_id="example"), data=SimpleNamespace(params=params))
    validate_reuse(run, data, joint_config, 2)
    for key, changed in [("dataset_sha256", "other-data"), ("learning_rate", "0.1"), ("n_jobs", "4")]:
        original = params[key]
        params[key] = changed
        with pytest.raises(ValueError, match="incompatible settings"):
            validate_reuse(run, data, joint_config, 2)
        params[key] = original
