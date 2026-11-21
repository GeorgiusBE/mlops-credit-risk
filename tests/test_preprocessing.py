"""Exercise preprocessing with independent, small raw tables and known results."""

from datetime import date
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

from data_preprocessing import feature_engineering as engineering
from data_preprocessing import model_input_validation as validation
from data_preprocessing import table_cleaning as cleaning
from data_preprocessing import table_joins as joins
from data_preprocessing.__main__ import main, make_parser
from data_preprocessing.common import PROJECT_ROOT, training_paths, write_parquet_atomic
from feature_selection import train_baseline as baseline
from feature_selection.select_features import build_feature_sets
from prediction_api import RawApplication, application_features


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def raw_dir(tmp_path):
    """Eight cases, sharded tables, secondary people, and unmatched details."""
    raw = tmp_path / "raw inputs"
    train = raw / "train"
    save(pd.DataFrame({
        "case_id": range(1, 9), "date_decision": ["2020-01-10"] * 8,
        "WEEK_NUM": [0, 0, 1, 1, 2, 2, 3, 3], "MONTH": [202001] * 8,
        "target": [0, 1] * 4,
    }), train / "train_base.parquet")
    static = pd.DataFrame({
        "case_id": range(1, 9), "annuity_780A": [100.] * 8,
        "credamount_770A": [500.] * 8,
    })
    save(static.iloc[:4], train / "train_static_0_0.parquet")
    save(static.iloc[4:], train / "train_static_0_1.parquet")
    save(pd.DataFrame({"case_id": range(1, 9), "noteM": ["", " \t", "OK"] + [None] * 5}),
         train / "train_static_cb_0.parquet")

    # The same-day/future records must be excluded from engineered date features.
    history_dates = ["2019-12-31", "2020-01-05", "2020-01-10", "2020-01-11", "2020-01-09", "2020-01-01"]
    history = pd.DataFrame({
        "case_id": [1, 1, 1, 1, 2, 999], "num_group1": [0, 1, 2, 3, 0, 0],
        "amountA": [10., 30., 50., 70., 90., 1000.],
        "labelM": ["A", "B", "A", "C", "B", "orphan"],
    })
    date_columns = {"debitcard": "openingdate_857D", "deposit": "openingdate_313D",
                    "applprev": "creationdate_885D", "credit_bureau_a": "lastupdate_1112D"}
    for source in (*joins.DEPTH1_SOURCES, "applprev", "credit_bureau_a", "credit_bureau_b"):
        frame = history.copy()
        frame[date_columns.get(source, "eventD")] = history_dates
        if source in {"applprev", "credit_bureau_a"}:
            save(frame.iloc[:3], train / f"train_{source}_1_0.parquet")
            save(frame.iloc[3:], train / f"train_{source}_1_1.parquet")
        else:
            save(frame, train / f"train_{source}_1.parquet")

    save(pd.DataFrame({
        "case_id": [1, 1, *range(2, 9)], "num_group1": [0, 1, *([0] * 7)],
        "mainoccupationinc_384A": [1000., 99999., 0., *([500.] * 6)],
        "incometype_1044T": ["salary", "secondary", *(["salary"] * 7)],
        "empl_industry_691L": ["A", "secondary", "B", "A", "B", "VAL", "A", "TEST", "B"],
        "empl_employedfrom_271D": ["2020-01-01", "1990-01-01", "2020-01-11", *([None] * 6)],
    }), train / "train_person_1.parquet")

    details = pd.DataFrame({
        "case_id": [1] * 4, "num_group1": [0, 0, 1, 99], "num_group2": [0, 1, 0, 0],
        "amountA": [10., 30., 100., 9999.], "kindM": ["A", "B", "C", "orphan"],
        "flag": [True, False, True, False], "ordinalT": [1, 3, 7, 99],
        "eventD": ["2020-01-01", "2020-01-03", "2020-01-07", "2020-01-09"],
    })
    for source in joins.PAIRED_SOURCES:
        save(details.iloc[:2], train / f"train_{source}_2_0.parquet")
        save(details.iloc[2:], train / f"train_{source}_2_1.parquet")
    return raw


def test_complete_pipeline_cli_and_training_compatibility(raw_dir, tmp_path):
    """A fresh process must produce model-ready files without a notebook kernel."""
    processed = tmp_path / "processed outputs"
    result = subprocess.run([
        sys.executable, "-m", "data_preprocessing", "--stage", "all",
        "--raw-dir", str(raw_dir), "--processed-dir", str(processed), "--threads", "1",
    ], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Saved 20 eligible features" in result.stdout
    for depth in range(3):
        table = pd.read_parquet(processed / f"train_base_depth{depth}.parquet")
        assert sorted(table.case_id) == list(range(1, 9))

    joined = pd.read_parquet(processed / "train_base_depth2.parquet").set_index("case_id")
    assert joined.loc[1, "debitcard_record_count"] == 4
    assert joined.loc[1, "debitcard_amountA_avg"] == 40
    assert joined.loc[1, "debitcard_amountA_max"] == 70
    assert joined.loc[1, "other_amountA_avg"] == 40
    assert "other_amountA_max" not in joined
    assert joined.loc[1, "debitcard_labelM_distinct_count"] == 3
    assert joined.loc[3, "debitcard_record_count"] == 0
    for source in joins.PAIRED_SOURCES:
        assert joined.loc[1, f"{source}_2_record_count"] == 3
        assert joined.loc[1, f"{source}_2_parent_count"] == 2
        assert joined.loc[1, f"{source}_2_max_records_per_parent"] == 2
        assert joined.loc[1, f"{source}_2_amountA_avg_across_parents"] == 60
        assert joined.loc[1, f"{source}_2_amountA_max"] == 100
        assert joined.loc[1, f"{source}_2_kindM_distinct_count_max_per_parent"] == 2
        assert joined.loc[1, f"{source}_2_flag_true_share_across_parents"] == .75
        assert joined.loc[1, f"{source}_2_ordinalT_min"] == 1
        assert joined.loc[1, f"{source}_2_ordinalT_max"] == 7
        assert joined.loc[2, f"{source}_2_record_count"] == 0
    assert joined.loc[1, "person_1_applicant_mainoccupationinc_384A"] == 1000

    cleaned = pd.read_parquet(processed / "train_base_cleaned.parquet").set_index("case_id")
    assert cleaned.loc[[1, 2], "static_cb_0_noteM"].isna().all()
    assert cleaned.loc[3, "static_cb_0_noteM"] == "OK"
    featured = pd.read_parquet(processed / "train_base_engineered.parquet").set_index("case_id")
    assert len(featured.columns) == len(cleaned.columns) + 15
    assert featured.loc[1, "annuity_to_reported_income"] == .1
    assert featured.loc[1, "credit_amount_to_reported_income"] == .5
    assert featured.loc[1, "employment_tenure_days"] == 9
    assert pd.isna(featured.loc[2, "annuity_to_reported_income"])
    assert pd.isna(featured.loc[2, "employment_tenure_days"])
    for prefix in ("debit_card_opening", "deposit_opening", "prior_application", "active_bureau_update"):
        assert featured.loc[1, f"{prefix}_earliest_days_ago"] == 10
        assert featured.loc[1, f"{prefix}_latest_days_ago"] == 5
        assert featured.loc[1, f"{prefix}_span_days"] == 5
        assert featured.loc[2, f"{prefix}_span_days"] == 0
        assert pd.isna(featured.loc[3, f"{prefix}_span_days"])

    inventory = pd.read_csv(processed / "model_feature_eligibility.csv")
    assert list(inventory) == ["column_name", "data_type", "role", "reason"]
    assert inventory.column_name.tolist() == ["case_id", *featured.columns]
    features = baseline.eligible_features(processed / "model_feature_eligibility.csv")
    assert len(features) == 20
    serving_features = [name for name in features if not name.startswith("deposit_opening_")]
    histories = ["2019-12-31", "2020-01-05", "2020-01-10", "2020-01-11", None]
    application = RawApplication.model_validate({
        "decision_date": "2020-01-10",
        "annuity_amount": 100,
        "credit_amount": 500,
        "employment_industry": "A",
        "income_type": "salary",
        "main_occupation_income": 1000,
        "employment_start_date": "2020-01-01",
        "debit_card_opening_dates": histories,
        "prior_application_dates": histories,
        "bureau_update_dates": histories,
    })
    pd.testing.assert_frame_equal(
        application_features(application)[serving_features],
        featured.loc[[1], serving_features].reset_index(drop=True),
        check_dtype=False,
    )
    config = baseline.load_config(baseline.DEFAULT_CONFIG)
    assert len(build_feature_sets(features, config)) == 8
    config["split"] = {"train_start": 0, "train_end": 1, "validation_end": 2, "test_end": 3}
    data = baseline.load_training_data(processed, config, 1)
    assert data.splits["held_out_test"] == [3]
    industry = "person_1_applicant_empl_industry_691L"
    assert data.train[industry].cat.categories.tolist() == ["A", "B"]
    assert data.validation[industry].isna().sum() == 1

    # Exercise Make's actual quoting and individual-stage dispatch on disposable data.
    quoted_dir = tmp_path / "it's processed"
    quoted_dir.mkdir()
    (quoted_dir / "train_base_engineered.parquet").write_bytes(
        (processed / "train_base_engineered.parquet").read_bytes()
    )
    result = subprocess.run([
        "make", "preprocess-validate", f"PYTHON={sys.executable}",
        f"RAW_DIR={raw_dir}", f"PROCESSED_DIR={quoted_dir}", "THREADS=1",
    ], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (quoted_dir / "model_feature_eligibility.csv").read_bytes() == (processed / "model_feature_eligibility.csv").read_bytes()


def valid_cases():
    return pd.DataFrame({
        "case_id": [1, 2], "date_decision": [date(2020, 1, 10)] * 2,
        "WEEK_NUM": [0, 1], "target": [0, 1],
    })


def test_cleaning_dates_infinities_and_quoted_blank_columns(tmp_path):
    cases = valid_cases()
    cases["date_decision"] = ["2020-01-10"] * 2
    cases["startD"] = ["2020-01-01", None]
    cases["amount"] = [np.inf, -np.inf]
    cases['odd"name'] = ["\t\n ", " keep spaces "]
    save(cases, tmp_path / "train_base_depth2.parquet")
    output = cleaning.run(tmp_path / "unused", tmp_path, 1)
    with duckdb.connect() as con:
        table = con.read_parquet(str(output))
        schema = dict(zip(table.columns, map(str, table.types)))
        assert schema["date_decision"] == schema["startD"] == "DATE"
        frame = table.order("case_id").df()
    assert frame.amount.isna().all()
    assert pd.isna(frame['odd"name'].iloc[0])
    assert frame['odd"name'].iloc[1] == " keep spaces "
    # A second clean pass has DATE columns already and no remaining dirty values.
    save(pd.read_parquet(output), tmp_path / "train_base_depth2.parquet")
    cleaning.run(None, tmp_path, 1)


def test_cleaning_without_optional_type_groups(tmp_path):
    save(valid_cases(), tmp_path / "train_base_depth2.parquet")
    output = cleaning.run(None, tmp_path, 1)
    pd.testing.assert_frame_equal(pd.read_parquet(output), valid_cases())


@pytest.mark.parametrize("column,values,message", [
    ("case_id", [1, 1], "non-null case_id"),
    ("case_id", [1, None], "non-null case_id"),
    ("target", [0, 2], "non-binary targets"),
    ("target", [0, None], "non-binary targets"),
    ("WEEK_NUM", [0, None], "missing WEEK_NUM"),
    ("date_decision", [date(2020, 1, 10), None], "missing date_decision"),
])
def test_cleaning_rejects_invalid_cases_without_replacing_output(tmp_path, column, values, message):
    cases = valid_cases()
    cases[column] = values
    save(cases, tmp_path / "train_base_depth2.parquet")
    output = tmp_path / "train_base_cleaned.parquet"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match=message):
        cleaning.run(None, tmp_path, 1)
    assert output.read_bytes() == b"previous output"


def test_cleaning_rejects_empty_input(tmp_path):
    save(valid_cases().iloc[:0], tmp_path / "train_base_depth2.parquet")
    with pytest.raises(ValueError, match="empty"):
        cleaning.run(None, tmp_path, 1)


@pytest.mark.parametrize("income,annuity,credit,tenure,expected", [
    (100., 10., 50., date(2020, 1, 1), (.1, .5, 9)),
    (100., 0., 0., date(2020, 1, 10), (0., 0., 0)),
    (0., 10., 50., date(2020, 1, 11), (None, None, None)),
    (-1., 10., 50., None, (None, None, None)),
    (None, 10., 50., None, (None, None, None)),
    (np.inf, 10., 50., None, (None, None, None)),
    (100., np.inf, -1., None, (None, None, None)),
    (100., -1., np.inf, None, (None, None, None)),
    (100., None, None, None, (None, None, None)),
])
def test_case_feature_domain_guards(income, annuity, credit, tenure, expected):
    with duckdb.connect() as con:
        cases = con.sql("""
            SELECT CAST(? AS DOUBLE) AS person_1_applicant_mainoccupationinc_384A,
                   CAST(? AS DOUBLE) AS static_0_annuity_780A,
                   CAST(? AS DOUBLE) AS static_0_credamount_770A,
                   CAST(? AS DATE) AS person_1_applicant_empl_employedfrom_271D,
                   DATE '2020-01-10' AS date_decision
        """, params=[income, annuity, credit, tenure])
        [actual] = engineering.add_case_features(cases).project(
            "annuity_to_reported_income, credit_amount_to_reported_income, employment_tenure_days"
        ).fetchall()
        assert actual == expected


def test_engineered_validation_rejects_bad_values_and_changed_grain():
    with duckdb.connect() as con:
        source = con.from_df(valid_cases())
        extra = ", ".join(f"1.0 AS feature_{i}" for i in range(15))
        featured = source.project(f"*, {extra}")
        assert engineering.validate_engineered_features(source, featured) == 2
        for expression in ("-1.0", "CAST('inf' AS DOUBLE)"):
            bad = featured.project(f"* REPLACE ({expression} AS feature_0)")
            with pytest.raises(ValueError, match="invalid engineered values"):
                engineering.validate_engineered_features(source, bad)
        with pytest.raises(ValueError, match="preserve"):
            engineering.validate_engineered_features(source, featured.filter("case_id = 1"))


def test_coverage_and_inventory_requirements():
    with duckdb.connect() as con:
        cases = con.from_df(pd.DataFrame({
            "WEEK_NUM": [0, 0, 1], "value": [1., None, None],
        }))
        inventory = pd.DataFrame({"column_name": ["WEEK_NUM", "value"], "role": ["metadata", "eligible"]})
        coverage = validation.weekly_feature_coverage(cases, inventory)
        assert coverage.coverage_pct.tolist() == [50., 0.]
        assert coverage.case_count.tolist() == [2, 1]
        summary = validation.summarize_eligible_coverage(coverage, inventory)
        assert summary.loc["value"].tolist() == [0., 50., 1.]
        for invalid in (cases.limit(0), cases.project("* REPLACE (NULL AS WEEK_NUM)")):
            with pytest.raises(ValueError, match="Weekly coverage"):
                validation.weekly_feature_coverage(invalid, inventory)
    with pytest.raises(ValueError, match="Missing expected columns"):
        validation.build_feature_inventory({"case_id": "INTEGER"})


@pytest.mark.parametrize("table", ["train_base.parquet", "train_static_0_0.parquet", "train_static_cb_0.parquet"])
def test_join_rejects_null_depth0_keys(raw_dir, tmp_path, table):
    path = raw_dir / "train" / table
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], "case_id"] = None
    save(frame, path)
    with pytest.raises(ValueError, match="non-null case_id"):
        joins.run(raw_dir, tmp_path / "out", 1)
    assert not (tmp_path / "out" / "train_base_depth0.parquet").exists()


def test_join_rejects_duplicate_keys_across_static_shards(raw_dir, tmp_path):
    path = raw_dir / "train" / "train_static_0_1.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], "case_id"] = 1
    save(frame, path)
    with pytest.raises(ValueError, match="non-null case_id"):
        joins.run(raw_dir, tmp_path / "out", 1)


def test_empty_static_table_preserves_base_cases(raw_dir, tmp_path):
    path = raw_dir / "train" / "train_static_cb_0.parquet"
    save(pd.read_parquet(path).iloc[:0], path)
    output = tmp_path / "depth0.parquet"
    joins.join_depth0(raw_dir / "train", output, 1)
    result = pd.read_parquet(output)
    assert sorted(result.case_id) == list(range(1, 9))
    assert result.static_cb_0_noteM.isna().all()


def test_history_discovery_precedence_and_missing_inputs(tmp_path):
    shard = tmp_path / "train_person_1_0.parquet"
    shard.touch()
    assert training_paths(tmp_path, "person", 1) == [shard]
    single = tmp_path / "train_person_1.parquet"
    single.touch()
    assert training_paths(tmp_path, "person", 1) == [single]
    with pytest.raises(FileNotFoundError, match="train_deposit_1"):
        training_paths(tmp_path, "deposit", 1)
    for stage in (cleaning, engineering, validation):
        with pytest.raises(FileNotFoundError, match="make preprocess-"):
            stage.run(tmp_path / "raw", tmp_path, 1)


def test_atomic_write_preserves_existing_output_on_failure(tmp_path):
    output = tmp_path / "result.parquet"
    output.write_bytes(b"previous output")

    class FailingRelation:
        def write_parquet(self, path, **options):
            Path(path).write_bytes(b"partial output")
            raise OSError("out of disk space")

    with pytest.raises(OSError, match="disk space"):
        write_parquet_atomic(FailingRelation(), output)
    assert output.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [output]


def test_cli_paths_validation_and_failure_messages(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    defaults = make_parser().parse_args([])
    assert defaults.raw_dir == PROJECT_ROOT / "dataset" / "raw"
    assert defaults.processed_dir == PROJECT_ROOT / "dataset" / "processed"
    assert defaults.stage == "all" and defaults.threads == 2
    with pytest.raises(SystemExit) as invalid_threads:
        main(["--threads", "0"])
    assert invalid_threads.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as missing_upstream:
        main(["--stage", "engineer", "--processed-dir", str(tmp_path)])
    assert missing_upstream.value.code == 1
    assert "make preprocess-clean" in capsys.readouterr().err


def test_make_dispatch_for_all_stages(tmp_path):
    for stage in ("all", "join", "clean", "engineer", "validate"):
        target = "preprocess" if stage == "all" else f"preprocess-{stage}"
        result = subprocess.run([
            "make", "-n", target, f"PYTHON={sys.executable}",
            f"RAW_DIR={tmp_path / 'raw'}", f"PROCESSED_DIR={tmp_path / 'processed'}", "THREADS=3",
        ], cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert f"--stage {stage}" in result.stdout
        assert "--threads '3'" in result.stdout


def test_import_has_no_processing_side_effects(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run([
        sys.executable, "-c",
        "from data_preprocessing import table_joins, table_cleaning, feature_engineering, model_input_validation",
    ], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not list(tmp_path.iterdir())
