"""Check paired temporal comparisons, reuse constraints, and final-test isolation."""

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from feature_selection import train_baseline as baseline
from feature_selection import confirm_features as confirmation
from feature_selection.confirm_features import (
    DEFAULT_CONFIG, configure_joint_groups, inherit_joint_plan, plan_confirmation, resolve_joint_study_id,
    select_confirmation_sets, summarize_windows, validate_reuse, window_config,
)


class FeatureConfirmationTest(unittest.TestCase):
    """Protect the joint and temporal checks from invalid comparisons."""

    def setUp(self):
        """Load the full plan without reading outcome data or contacting MLflow."""
        self.raw_config = baseline.load_config(DEFAULT_CONFIG)
        self.groups = ["deposit_history", "affordability_ratios", "debit_card_history"]
        self.config = configure_joint_groups(self.raw_config, self.groups)
        self.triple = "without_affordability_ratios__and__debit_card_history__and__deposit_history"
        self.features = [name for columns in self.config["feature_groups"].values() for name in columns]

    def test_joint_groups_require_explicit_valid_choices(self):
        """No default shortlist may silently advance groups from screening."""
        self.assertEqual(self.raw_config["candidate_groups"], [])
        self.assertNotIn("ablations", self.raw_config)
        for groups in ([], ["deposit_history"], ["deposit_history", "deposit_history"],
                       ["deposit_history", "unknown"], "deposit_history"):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                configure_joint_groups(self.raw_config, groups)
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["confirm_features.py", "--stage", "joint"]))
            stack.enter_context(patch.object(baseline, "load_config", return_value=self.raw_config))
            stack.enter_context(patch.object(baseline, "eligible_features", return_value=self.features))
            load = stack.enter_context(patch.object(baseline, "load_training_data"))
            tracking = stack.enter_context(patch.object(confirmation.mlflow, "set_tracking_uri"))
            error = stack.enter_context(redirect_stderr(io.StringIO()))
            with self.assertRaises(SystemExit) as exit_status:
                confirmation.main()
            self.assertEqual(exit_status.exception.code, 2)
            self.assertIn("Choose at least two groups", error.getvalue())
            load.assert_not_called()
            tracking.assert_not_called()

    def test_joint_combinations_follow_arbitrary_human_group_choices(self):
        """Generate only chosen subsets, with stable labels and no three-group limit."""
        groups = ["loan_amounts", "applicant_information"]
        config = configure_joint_groups(self.raw_config, groups)
        sets = plan_confirmation(config, self.features)
        self.assertEqual(len(config["ablations"]), 2)  # Fresh baseline and one pair.
        self.assertEqual(len(sets), 4)  # Plus the reused single removals.
        pair = sets["without_applicant_information__and__loan_amounts"]
        self.assertEqual(set(pair["excluded_groups"]), set(groups))
        self.assertEqual(len(pair["features"]), 14)
        reversed_config = configure_joint_groups(self.raw_config, list(reversed(groups)))
        self.assertEqual(config["ablations"], reversed_config["ablations"])
        four = configure_joint_groups(self.raw_config, [*self.groups, "loan_amounts"])
        self.assertEqual(len(four["ablations"]), 12)
        self.assertEqual(len(plan_confirmation(four, self.features)), 16)
        self.assertEqual(self.raw_config["candidate_groups"], [])
        self.assertNotIn("ablations", self.raw_config)

    def test_reviewed_plan_preserves_choices_and_legacy_labels_without_shortlisting(self):
        """Reuse previous group decisions and names, but require a fresh shortlist."""
        reviewed = copy.deepcopy(self.config)
        reviewed["ablations"]["legacy_triple_name"] = reviewed["ablations"].pop(self.triple)
        reviewed["confirmation"]["selected_configurations"] = ["legacy_triple_name"]
        reviewed["screening_parent_run_id"] = "reviewed-screening"
        reviewed["model"]["learning_rate"] = 0.1
        inherited = inherit_joint_plan(self.raw_config, reviewed)
        self.assertEqual(inherited["candidate_groups"], self.groups)
        self.assertIn("legacy_triple_name", plan_confirmation(inherited, self.features))
        self.assertEqual(inherited["confirmation"]["selected_configurations"], [])
        self.assertEqual(inherited["model"], self.raw_config["model"])
        self.assertEqual(inherited["screening_parent_run_id"], "reviewed-screening")
        self.assertEqual(inherit_joint_plan(self.raw_config, reviewed, inherit_screening=False)["screening_parent_run_id"], self.raw_config["screening_parent_run_id"])
        inherit_joint_plan(self.raw_config, reviewed, list(reversed(self.groups)))
        with self.assertRaisesRegex(ValueError, "differ from the reviewed"):
            inherit_joint_plan(self.raw_config, reviewed, ["loan_amounts", "deposit_history"])

    def test_all_eight_patterns_and_final_test_are_explicit(self):
        """Account for the baseline, three singles, three pairs, and triple drop."""
        sets = plan_confirmation(self.config, self.features)
        self.assertEqual(len(sets), 8)
        self.assertEqual(len(sets["without_debit_card_history__and__deposit_history"]["features"]), 14)
        self.assertEqual(len(sets[self.triple]["features"]), 12)
        for window in self.config["confirmation"]["windows"]:
            current = window_config(self.config, window, {"all_eligible": []})
            self.assertEqual(current["split"]["test_start"], 78)
            self.assertEqual(current["split"]["test_end"], 91)
            self.assertLess(current["split"]["validation_end"], 78)
        incomplete = copy.deepcopy(self.config)
        del incomplete["ablations"][self.triple]
        with self.assertRaisesRegex(ValueError, "every joint"):
            plan_confirmation(incomplete, self.features)

    def test_reject_overlapping_windows_and_final_test_validation(self):
        """A confirmation window must not consume another validation window or test."""
        bad = copy.deepcopy(self.config)
        bad["confirmation"]["windows"][1]["train_end"] = 40
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            plan_confirmation(bad, self.features)
        with self.assertRaisesRegex(ValueError, "precede the fixed final test"):
            window_config(self.config, {"train_end": 63, "validation_end": 78}, {"all_eligible": []})

    def test_loader_skips_future_gap_and_retains_original_test(self):
        """Unused future labels and final test labels must stay out of loaded frames."""
        config = copy.deepcopy(self.config)
        config["split"] = {"train_start": 0, "train_end": 0, "validation_end": 1, "test_start": 3, "test_end": 3}
        source = pd.DataFrame({
            "WEEK_NUM": [0, 0, 1, 1, 2, 2, 3, 3],
            "target": [0, 1, 0, 1, 777, 777, 999, 999],
            "amount": list(range(8)),
        })
        with TemporaryDirectory() as directory:
            path = Path(directory)
            source.to_parquet(path / "train_base_engineered.parquet", index=False)
            pd.DataFrame({"column_name": ["amount"], "role": ["eligible"]}).to_csv(path / "model_feature_eligibility.csv", index=False)
            data = baseline.load_training_data(path, config, 1)
            self.assertEqual(data.splits, {"train": [0], "validation": [1], "held_out_test": [3], "unused_future": [2]})
            self.assertEqual(data.train["amount"].tolist(), [0, 1])
            self.assertEqual(data.validation["amount"].tolist(), [2, 3])

    def test_paired_deltas_use_each_windows_baseline(self):
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
        self.assertAlmostEqual(rows.loc["reduced", "mean_delta_gini_stability"], 0.005)
        self.assertAlmostEqual(rows.loc["reduced", "worst_delta_gini_stability"], -0.01)
        self.assertEqual(rows.loc["reduced", "windows_improved"], 1)
        self.assertEqual(rows.loc["full", "mean_delta_gini_stability"], 0)
        with self.assertRaisesRegex(ValueError, "every confirmation window"):
            summarize_windows({"early": first, "late": second.iloc[:1]}, "full")

    def test_manual_shortlist_has_no_automatic_count_or_ranking(self):
        """Keep any valid human choices and their order, even beyond the old top two."""
        self.config["confirmation"]["candidate_count"] = 1  # Legacy settings cannot select anything.
        sets = plan_confirmation(self.config, self.features)
        names = [self.triple, "without_affordability_ratios", "without_deposit_history"]
        selected = select_confirmation_sets(sets, names)
        self.assertEqual(list(selected), ["all_eligible", *names])
        self.assertEqual(select_confirmation_sets(sets, [], required=False), {})
        with self.assertRaisesRegex(ValueError, "Choose configurations"):
            select_confirmation_sets(sets, [])
        for invalid in (["not_a_configuration"], [names[0], names[0]], ["all_eligible"]):
            with self.assertRaises(ValueError):
                select_confirmation_sets(sets, invalid)

    def test_temporal_cli_requires_human_choice_and_review_before_data_loading(self):
        """Missing choices or review ID must fail before MLflow calls or training."""
        cases = [
            (["--joint-study-id", "reviewed-study"], "Choose configurations"),
            (["--candidates", "without_deposit_history"], "requires JOINT_STUDY_ID"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments), ExitStack() as stack:
                stack.enter_context(patch("sys.argv", ["confirm_features.py", *arguments]))
                stack.enter_context(patch.object(baseline, "load_config", return_value=copy.deepcopy(self.config)))
                stack.enter_context(patch.object(baseline, "eligible_features", return_value=self.features))
                load = stack.enter_context(patch.object(baseline, "load_training_data"))
                tracking = stack.enter_context(patch.object(confirmation.mlflow, "set_tracking_uri"))
                error = stack.enter_context(redirect_stderr(io.StringIO()))
                with self.assertRaises(SystemExit) as exit_status:
                    confirmation.main()
                self.assertEqual(exit_status.exception.code, 2)
                self.assertIn(message, error.getvalue())
                load.assert_not_called()
                tracking.assert_not_called()

    def test_joint_review_stops_and_temporal_uses_only_explicit_choices(self):
        """Exercise both stages with logging/fits mocked, including the review pause."""
        sets = plan_confirmation(self.config, self.features)
        results = {
            name: {
                "run_id": f"run-{name}", "model_uri": f"models:/{name}",
                "feature_count": len(definition["features"]), "best_iteration": 10,
                "metrics": {"roc_auc": 0.65, "gini_stability": 0.3, "average_precision": 0.2, "log_loss": 0.5},
                "weekly": pd.DataFrame({"WEEK_NUM": [64, 65], "gini": [0.2, 0.3]}),
            }
            for name, definition in sets.items()
        }
        data = SimpleNamespace(features=self.features, fingerprints={"dataset_sha256": "same-data"}, splits={"train": list(range(64)), "validation": list(range(64, 78)), "held_out_test": list(range(78, 92))})
        chosen = [self.triple, "without_affordability_ratios"]
        for stage in ("joint", "temporal"):
            with self.subTest(stage=stage), ExitStack() as stack:
                directory = stack.enter_context(TemporaryDirectory())
                tracking_path = Path(directory) / "test.db"
                previous_uri = confirmation.mlflow.get_tracking_uri()
                confirmation.mlflow.set_tracking_uri(f"sqlite:///{tracking_path}")
                stack.callback(confirmation.mlflow.set_tracking_uri, previous_uri)
                arguments = ["confirm_features.py", "--stage", stage]
                if stage == "temporal":
                    arguments += ["--joint-study-id", "reviewed-study", "--candidates", *chosen]
                else:
                    arguments += ["--candidate-groups", *self.groups]
                stack.enter_context(patch("sys.argv", arguments))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(baseline, "load_config", return_value=copy.deepcopy(self.raw_config)))
                stack.enter_context(patch.object(baseline, "eligible_features", return_value=self.features))
                loader = stack.enter_context(patch.object(baseline, "load_training_data", return_value=data))
                stack.enter_context(patch.object(confirmation, "load_study_results", return_value=results))
                stack.enter_context(patch.object(confirmation, "read_json_artifact", return_value=copy.deepcopy(self.config)))
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
                self.assertEqual(recorded["available_feature_sets.json"], sets)
                self.assertEqual(recorded["joint_group_selection.json"]["candidate_groups"], self.groups)
                self.assertEqual(recorded["joint_group_selection.json"]["selection_method"], "human")
                self.assertEqual(recorded["joint_group_selection.json"]["selection_source"], "command_line" if stage == "joint" else "reviewed_study")
                if stage == "joint":
                    loader.assert_called_once()
                    fits.assert_called_once()
                    confirmation.log_joint_plot.assert_called_once()
                    confirmation.log_plots.assert_not_called()
                    self.assertNotIn("confirmation_candidates.json", recorded)
                    confirmation.mlflow.set_tag.assert_called_with("result_status", "awaiting_human_shortlist")
                else:
                    self.assertEqual(fits.call_count, 2)  # Earlier windows; latest reuses models.
                    for call in fits.call_args_list:
                        self.assertEqual(list(call.args[1]["ablations"]), ["all_eligible", *chosen])
                    self.assertEqual(recorded["confirmation_selection.json"]["selected_configurations"], chosen)
                    self.assertEqual(recorded["confirmation_selection.json"]["selection_method"], "human")
                    self.assertNotIn("recommendation.json", recorded)
                    confirmation.mlflow.set_tag.assert_called_with("result_status", "awaiting_human_feature_decision")
                self.assertFalse(tracking_path.exists(), "Mocked workflow must not create real tracking records.")

    def test_resolve_completed_review_to_fitted_joint_source(self):
        """A review's manifest must resolve to a checked fitted joint study."""
        client = MagicMock()
        client.get_run.return_value.info.status = "FINISHED"
        client.list_artifacts.return_value = [SimpleNamespace(path="source_runs.json")]
        with patch.object(confirmation, "read_json_artifact", return_value={"joint_study_id": "fitted-study"}):
            self.assertEqual(resolve_joint_study_id(client, "reviewed-study"), "fitted-study")
        client.list_artifacts.return_value = []
        self.assertEqual(resolve_joint_study_id(client, "fitted-study"), "fitted-study")

    def test_refuse_reuse_after_dataset_or_parameter_change(self):
        """Old screening results require matching data, settings, and completion."""
        data = SimpleNamespace(fingerprints={"dataset_sha256": "data", "feature_inventory_sha256": "inventory"}, train=[1, 2], validation=[1, 2])
        params = {key: str(value) for key, value in {
            **self.config["model"], **data.fingerprints, "n_jobs": 2,
            "early_stopping_rounds": 40, "train_cases": 2, "validation_cases": 2,
        }.items()}
        run = SimpleNamespace(info=SimpleNamespace(status="FINISHED", run_id="example"), data=SimpleNamespace(params=params))
        validate_reuse(run, data, self.config, 2)
        for key, changed in [("dataset_sha256", "other-data"), ("learning_rate", "0.1"), ("n_jobs", "4")]:
            original = params[key]
            params[key] = changed
            with self.assertRaisesRegex(ValueError, "incompatible settings"):
                validate_reuse(run, data, self.config, 2)
            params[key] = original


if __name__ == "__main__":
    unittest.main()
