"""Guard feature membership, temporal isolation, and ablation comparisons."""

import copy
import json

import pandas as pd
import pytest

from feature_selection import train_baseline as baseline
from feature_selection.select_features import build_feature_sets, comparison_table


@pytest.fixture
def selection_config():
    """Load fresh configuration so each test can edit it independently."""
    return baseline.load_config(baseline.DEFAULT_CONFIG)


@pytest.fixture
def eligible_features(selection_config):
    return [column for columns in selection_config["feature_groups"].values() for column in columns]


def test_default_ablation_membership_and_order(selection_config, eligible_features):
    """Drop exactly one whole group and preserve the surviving column order."""
    features = list(reversed(eligible_features))
    sets = build_feature_sets(features, selection_config)
    assert len(features) == 20
    assert len(sets) == 8
    assert sets["all_eligible"]["features"] == features
    for group, columns in selection_config["feature_groups"].items():
        actual = sets[f"without_{group}"]["features"]
        assert actual == [name for name in features if name not in columns]


def test_reject_overlapping_unassigned_and_ineligible_columns(selection_config, eligible_features):
    """An eligibility change must be reflected explicitly in the groups."""
    overlapping = copy.deepcopy(selection_config)
    overlapping["feature_groups"]["loan_amounts"].append(eligible_features[0])
    with pytest.raises(ValueError, match="overlaps"):
        build_feature_sets(eligible_features, overlapping)
    with pytest.raises(ValueError, match="Unassigned"):
        build_feature_sets([*eligible_features, "new_feature"], selection_config)
    invalid = copy.deepcopy(selection_config)
    invalid["feature_groups"]["loan_amounts"].append("target")
    with pytest.raises(ValueError, match="unknown/ineligible"):
        build_feature_sets(eligible_features, invalid)


def test_combined_exclusions_and_reference_first(selection_config, eligible_features):
    """Support later combination checks without accidentally changing the reference."""
    selection_config["ablations"] = {
        "without_amounts_and_ratios": ["loan_amounts", "affordability_ratios"],
        "all_eligible": [],
    }
    sets = build_feature_sets(eligible_features, selection_config)
    assert next(iter(sets)) == "all_eligible"
    assert len(sets["without_amounts_and_ratios"]["features"]) == 16
    selection_config["ablations"]["another_reference"] = []
    with pytest.raises(ValueError, match="Exactly one"):
        build_feature_sets(eligible_features, selection_config)


def test_load_only_training_and_validation_with_training_categories(selection_config, tmp_path):
    """Test-only values and validation-only categories must not enter training."""
    config = copy.deepcopy(selection_config)
    config["split"] = {"train_start": 0, "train_end": 1, "validation_end": 2, "test_end": 3}
    source = pd.DataFrame({
        "WEEK_NUM": [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4,
        "target": [0, 1, 0, 1] * 3 + [999] * 4,
        "amount": list(range(12)) + [99999] * 4,
        "industry": ["A", "B", "A", "B"] * 2 + ["validation_only", None, "A", "B"] + ["test_only"] * 4,
    })
    source.to_parquet(tmp_path / "train_base_engineered.parquet", index=False)
    inventory = pd.DataFrame({"column_name": ["amount", "industry"], "role": ["eligible", "eligible"]})
    inventory.to_csv(tmp_path / "model_feature_eligibility.csv", index=False)
    data = baseline.load_training_data(tmp_path, config, 1)
    assert data.train["WEEK_NUM"].unique().tolist() == [0, 1]
    assert data.validation["WEEK_NUM"].unique().tolist() == [2]
    assert data.splits["held_out_test"] == [3]
    assert data.train["industry"].cat.categories.tolist() == ["A", "B"]
    assert data.validation["industry"].dtype == data.train["industry"].dtype
    assert data.validation["industry"].iloc[:2].isna().all()
    assert not data.train["target"].eq(999).any()
    assert not data.validation["amount"].eq(99999).any()
    with pytest.raises(ValueError, match="distinct subset"):
        baseline.train_and_log(data, config, ["target"], 1, "invalid")


def test_reject_label_as_eligible_and_overlapping_splits(selection_config, tmp_path):
    """Reject target leakage and invalid temporal boundaries before fitting."""
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame({"column_name": ["target"], "role": ["eligible"]}).to_csv(inventory, index=False)
    with pytest.raises(ValueError, match="metadata and target"):
        baseline.eligible_features(inventory)
    config_path = tmp_path / "config.json"
    selection_config["split"]["validation_end"] = selection_config["split"]["train_end"]
    config_path.write_text(json.dumps(selection_config))
    with pytest.raises(ValueError, match="non-overlapping"):
        baseline.load_config(config_path)


def test_comparison_delta_sign_and_weekly_consistency():
    """Positive deltas indicate improvement; count weekly gains separately."""
    reference = {
        "run_id": "reference", "model_uri": "models:/reference", "feature_count": 20, "best_iteration": 100,
        "metrics": {"roc_auc": 0.65, "gini_stability": 0.20},
        "weekly": pd.DataFrame({"WEEK_NUM": [64, 65], "gini": [0.2, 0.3]}),
    }
    reduced = {
        **reference, "run_id": "reduced", "feature_count": 18,
        "metrics": {"roc_auc": 0.66, "gini_stability": 0.19},
        "weekly": pd.DataFrame({"WEEK_NUM": [64, 65], "gini": [0.21, 0.29]}),
    }
    sets = {"full": {"excluded_groups": []}, "reduced": {"excluded_groups": ["loan_amounts"]}}
    table = comparison_table({"full": reference, "reduced": reduced}, sets).set_index("configuration")
    assert table.loc["full", "delta_gini_stability"] == 0
    assert table.loc["reduced", "delta_gini_stability"] == pytest.approx(-0.01, rel=0, abs=5e-8)
    assert table.loc["reduced", "delta_roc_auc"] == pytest.approx(0.01, rel=0, abs=5e-8)
    assert table.loc["reduced", "weeks_better_than_baseline"] == 1
    assert table.loc["reduced", "weeks_worse_than_baseline"] == 1
    assert table.loc["reduced", "weeks_compared"] == 2
