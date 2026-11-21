"""Add affordability ratios, employment tenure, and pre-decision history dates."""

from pathlib import Path

import duckdb

from .common import (
    check_training_counts,
    quote_identifier as quote,
    read_training_history,
    require_file,
    write_parquet_atomic,
)

HISTORY_DATE_SOURCES = (
    ("debitcard", "openingdate_857D", "debit_card_opening"),
    ("deposit", "openingdate_313D", "deposit_opening"),
    ("applprev", "creationdate_885D", "prior_application"),
    ("credit_bureau_a", "lastupdate_1112D", "active_bureau_update"),
)


def add_case_features(cases):
    """Derive two guarded ratios and nonnegative employment tenure in days."""
    return cases.project("""
        *,
        CASE
            WHEN isfinite(person_1_applicant_mainoccupationinc_384A)
                 AND person_1_applicant_mainoccupationinc_384A > 0
                 AND isfinite(static_0_annuity_780A)
                 AND static_0_annuity_780A >= 0
            THEN CAST(static_0_annuity_780A AS DOUBLE)
                 / person_1_applicant_mainoccupationinc_384A
        END AS annuity_to_reported_income,
        CASE
            WHEN isfinite(person_1_applicant_mainoccupationinc_384A)
                 AND person_1_applicant_mainoccupationinc_384A > 0
                 AND isfinite(static_0_credamount_770A)
                 AND static_0_credamount_770A >= 0
            THEN CAST(static_0_credamount_770A AS DOUBLE)
                 / person_1_applicant_mainoccupationinc_384A
        END AS credit_amount_to_reported_income,
        CASE
            WHEN person_1_applicant_empl_employedfrom_271D <= date_decision
            THEN date_diff('day', person_1_applicant_empl_employedfrom_271D, date_decision)
        END AS employment_tenure_days
    """)


def summarize_historical_dates(cases, history, date_column, feature_prefix):
    """Summarize only events strictly before each case's decision date."""
    decisions = cases.project("case_id, date_decision")
    events = history.project(f"case_id, CAST({date_column} AS DATE) AS event_date")
    days_before_decision = (
        events.join(decisions, "case_id")
        .filter("event_date < date_decision")
        .project("case_id, date_diff('day', event_date, date_decision) AS days_ago")
    )
    return days_before_decision.aggregate(f"""
        case_id,
        max(days_ago) AS {feature_prefix}_earliest_days_ago,
        min(days_ago) AS {feature_prefix}_latest_days_ago,
        max(days_ago) - min(days_ago) AS {feature_prefix}_span_days
    """, "case_id")


def engineer_features(con, cleaned, train_dir):
    """Return the original columns plus the 15 engineered features."""
    featured = add_case_features(cleaned)
    for source, date_column, feature_prefix in HISTORY_DATE_SOURCES:
        history = read_training_history(con, train_dir, source, 1)
        summary = summarize_historical_dates(cleaned, history, date_column, feature_prefix)
        featured = featured.join(summary, "case_id", how="left")
    return featured


def validate_engineered_features(cleaned, featured):
    """Check row preservation and engineered value domains; report coverage."""
    [(expected_rows,)] = cleaned.aggregate("count(*)").fetchall()
    cleaned_columns = set(cleaned.columns)
    engineered_columns = [name for name in featured.columns if name not in cleaned_columns]
    if len(engineered_columns) != 15:
        raise ValueError(f"Expected 15 new engineered columns; found {len(engineered_columns)}.")
    invalid_checks = " OR ".join(
        f"NOT isfinite({quote(name)}) OR {quote(name)} < 0" for name in engineered_columns
    )
    coverage_counts = ", ".join(f"count({quote(name)})" for name in engineered_columns)
    [(rows, unique_ids, invalid_targets, missing_weeks, missing_dates, invalid_features, *counts)] = featured.aggregate(f"""
        count(*), count(DISTINCT case_id),
        count(*) FILTER (WHERE target IS NULL OR target NOT IN (0, 1)),
        count(*) FILTER (WHERE WEEK_NUM IS NULL),
        count(*) FILTER (WHERE date_decision IS NULL),
        count(*) FILTER (WHERE {invalid_checks}),
        {coverage_counts}
    """).fetchall()
    check_training_counts(rows, unique_ids, invalid_targets, missing_weeks, missing_dates, expected_rows=expected_rows)
    if invalid_features:
        raise ValueError(f"Found {invalid_features:,} rows with invalid engineered values.")
    for name, count in zip(engineered_columns, counts):
        print(f"{name}: {count:,} of {rows:,} cases ({count / rows:.1%})")
    return rows


def run(raw_dir, processed_dir, threads=2):
    processed_dir = Path(processed_dir)
    input_path = require_file(processed_dir / "train_base_cleaned.parquet", "Run the clean stage: make preprocess-clean.")
    output_path = processed_dir / "train_base_engineered.parquet"
    with duckdb.connect(config={"threads": threads, "preserve_insertion_order": False}) as con:
        cleaned = con.read_parquet(str(input_path))
        featured = engineer_features(con, cleaned, Path(raw_dir) / "train")
        rows = validate_engineered_features(cleaned, featured)
        write_parquet_atomic(featured, output_path, row_group_size=32768)
    print(f"Saved {rows:,} cases to {output_path.resolve()}", flush=True)
    return output_path

