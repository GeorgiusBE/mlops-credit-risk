"""Guard feature membership, temporal isolation, and ablation comparisons."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from feature_selection import train_baseline as baseline
from feature_selection.select_features import build_feature_sets, comparison_table


class FeatureSelectionTest(unittest.TestCase):
    """Check the decisions that could invalidate a feature-selection study."""

    def setUp(self):
        """Load the shared experiment configuration for each independent check."""
        self.config = baseline.load_config(baseline.DEFAULT_CONFIG)
        self.features = [column for columns in self.config["feature_groups"].values() for column in columns]

    def test_default_ablation_membership_and_order(self):
        """Drop exactly one whole group and preserve the surviving column order."""
        features = list(reversed(self.features))
        sets = build_feature_sets(features, self.config)
        self.assertEqual(len(features), 20)
        self.assertEqual(len(sets), 8)
        self.assertEqual(sets["all_eligible"]["features"], features)
        for group, columns in self.config["feature_groups"].items():
            actual = sets[f"without_{group}"]["features"]
            self.assertEqual(actual, [name for name in features if name not in columns])

    def test_reject_overlapping_unassigned_and_ineligible_columns(self):
        """An eligibility change must be reflected explicitly in the groups."""
        overlapping = copy.deepcopy(self.config)
        overlapping["feature_groups"]["loan_amounts"].append(self.features[0])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_feature_sets(self.features, overlapping)
        with self.assertRaisesRegex(ValueError, "Unassigned"):
            build_feature_sets([*self.features, "new_feature"], self.config)
        invalid = copy.deepcopy(self.config)
        invalid["feature_groups"]["loan_amounts"].append("target")
        with self.assertRaisesRegex(ValueError, "unknown/ineligible"):
            build_feature_sets(self.features, invalid)

    def test_combined_exclusions_and_reference_first(self):
        """Support later combination checks without accidentally changing the reference."""
        self.config["ablations"] = {
            "without_amounts_and_ratios": ["loan_amounts", "affordability_ratios"],
            "all_eligible": [],
        }
        sets = build_feature_sets(self.features, self.config)
        self.assertEqual(next(iter(sets)), "all_eligible")
        self.assertEqual(len(sets["without_amounts_and_ratios"]["features"]), 16)
        self.config["ablations"]["another_reference"] = []
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            build_feature_sets(self.features, self.config)

    def test_load_only_training_and_validation_with_training_categories(self):
        """Test-only values and validation-only categories must not enter training."""
        config = copy.deepcopy(self.config)
        config["split"] = {"train_start": 0, "train_end": 1, "validation_end": 2, "test_end": 3}
        source = pd.DataFrame({
            "WEEK_NUM": [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4,
            "target": [0, 1, 0, 1] * 3 + [999] * 4,
            "amount": list(range(12)) + [99999] * 4,
            "industry": ["A", "B", "A", "B"] * 2 + ["validation_only", None, "A", "B"] + ["test_only"] * 4,
        })
        with TemporaryDirectory() as directory:
            processed = Path(directory)
            source.to_parquet(processed / "train_base_engineered.parquet", index=False)
            inventory = pd.DataFrame({"column_name": ["amount", "industry"], "role": ["eligible", "eligible"]})
            inventory.to_csv(processed / "model_feature_eligibility.csv", index=False)
            data = baseline.load_training_data(processed, config, 1)
            self.assertEqual(data.train["WEEK_NUM"].unique().tolist(), [0, 1])
            self.assertEqual(data.validation["WEEK_NUM"].unique().tolist(), [2])
            self.assertEqual(data.splits["held_out_test"], [3])
            self.assertEqual(data.train["industry"].cat.categories.tolist(), ["A", "B"])
            self.assertEqual(data.validation["industry"].dtype, data.train["industry"].dtype)
            self.assertTrue(data.validation["industry"].iloc[:2].isna().all())
            self.assertFalse(data.train["target"].eq(999).any())
            self.assertFalse(data.validation["amount"].eq(99999).any())
            with self.assertRaisesRegex(ValueError, "distinct subset"):
                baseline.train_and_log(data, config, ["target"], 1, "invalid")

    def test_reject_label_as_eligible_and_overlapping_splits(self):
        """Reject target leakage and invalid temporal boundaries before fitting."""
        with TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.csv"
            pd.DataFrame({"column_name": ["target"], "role": ["eligible"]}).to_csv(inventory, index=False)
            with self.assertRaisesRegex(ValueError, "metadata and target"):
                baseline.eligible_features(inventory)
            config_path = Path(directory) / "config.json"
            self.config["split"]["validation_end"] = self.config["split"]["train_end"]
            config_path.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                baseline.load_config(config_path)

    def test_comparison_delta_sign_and_weekly_consistency(self):
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
        self.assertEqual(table.loc["full", "delta_gini_stability"], 0)
        self.assertAlmostEqual(table.loc["reduced", "delta_gini_stability"], -0.01)
        self.assertAlmostEqual(table.loc["reduced", "delta_roc_auc"], 0.01)
        self.assertEqual(table.loc["reduced", "weeks_better_than_baseline"], 1)
        self.assertEqual(table.loc["reduced", "weeks_worse_than_baseline"], 1)
        self.assertEqual(table.loc["reduced", "weeks_compared"], 2)


if __name__ == "__main__":
    unittest.main()
