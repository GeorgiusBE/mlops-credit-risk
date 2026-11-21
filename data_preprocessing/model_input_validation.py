"""Build the fixed feature eligibility inventory and report coverage by week."""

from pathlib import Path

import duckdb
import pandas as pd

from .common import quote_identifier as quote, require_file, write_csv_atomic


def build_feature_inventory(schema):
    """Assign roles in source-column order, preserving the notebook's decisions."""
    metadata = {
        "case_id": "Case key for joining and tracing predictions",
        "date_decision": "Decision date for chronological validation and audit",
        "WEEK_NUM": "Chronological split and weekly evaluation group",
        "MONTH": "Calendar grouping for audit",
    }

    eligible = {
        "person_1_applicant_mainoccupationinc_384A": "Applicant's reported main income",
        "person_1_applicant_incometype_1044T": "Applicant's reported income type",
        "person_1_applicant_empl_industry_691L": "Applicant's employment industry",
        "employment_tenure_days": "Tenure calculated from the applicant's employment start",
        "static_0_annuity_780A": "Recorded annuity. Scoring-time availability is unverified.",
        "static_0_credamount_770A": "Recorded credit amount. Scoring-time availability is unverified.",
        "annuity_to_reported_income": "Depends on whether the recorded annuity was known at scoring",
        "credit_amount_to_reported_income": "Depends on whether the recorded credit amount was known at scoring",
    }
    history_date_reasons = {
        "debit_card_opening": "Opening history dated before the decision",
        "deposit_opening": "Opening history dated before the decision",
        "prior_application": "Event date passes the cutoff. Availability is unverified.",
        "active_bureau_update": "Event date passes the cutoff. Availability is unverified.",
    }
    for prefix, reason in history_date_reasons.items():
        for suffix in ("earliest_days_ago", "latest_days_ago", "span_days"):
            eligible[f"{prefix}_{suffix}"] = reason

    required = set(metadata) | {"target"} | set(eligible)
    missing = sorted(required - schema.keys())
    if missing:
        raise ValueError(f"Missing expected columns from the engineer stage: {', '.join(missing)}")

    rows = []
    for name, dtype in schema.items():
        if name in metadata:
            role, reason = "metadata", metadata[name]
        elif name == "target":
            role, reason = "label", "Observed default outcome"
        elif dtype == "DATE" or dtype.startswith("TIMESTAMP"):
            role, reason = "excluded", "Raw date retained for audit"
        elif name in eligible:
            role, reason = "eligible", eligible[name]
        else:
            role, reason = "review", "Review definition and availability at scoring"

        rows.append((name, dtype, role, reason))

    feature_inventory = pd.DataFrame(rows, columns=["column_name", "data_type", "role", "reason"])
    return feature_inventory


def weekly_feature_coverage(engineered, feature_inventory):
    """Return non-null coverage by week for every non-metadata, non-label field."""
    columns = feature_inventory.loc[
        ~feature_inventory["role"].isin(["metadata", "label"]), "column_name"
    ]
    counts = ", ".join(f"count({quote(name)}) AS {quote(name)}" for name in columns)
    weekly_counts = engineered.aggregate(
        f"WEEK_NUM, count(*) AS case_count, {counts}", "WEEK_NUM"
    ).order("WEEK_NUM").df()
    if weekly_counts.empty or weekly_counts["WEEK_NUM"].isna().any():
        raise ValueError("Weekly coverage requires a nonempty table with a WEEK_NUM for every case.")
    coverage = weekly_counts.melt(
        id_vars=["WEEK_NUM", "case_count"], var_name="column_name", value_name="non_null_cases",
    )
    coverage["coverage_pct"] = 100 * coverage["non_null_cases"] / coverage["case_count"]
    return coverage


def summarize_eligible_coverage(coverage, inventory):
    eligible_features = inventory.loc[inventory["role"].eq("eligible"), "column_name"]
    eligible_coverage = coverage.loc[coverage["column_name"].isin(eligible_features)]
    return eligible_coverage.groupby("column_name").agg(
        min_coverage_pct=("coverage_pct", "min"),
        max_coverage_pct=("coverage_pct", "max"),
        weeks_without_values=("non_null_cases", lambda counts: counts.eq(0).sum()),
    )


def run(raw_dir, processed_dir, threads=2):
    """Save the inventory after checking required columns and weekly coverage."""
    processed_dir = Path(processed_dir)
    input_path = require_file(processed_dir / "train_base_engineered.parquet", "Run the engineer stage: make preprocess-engineer.")
    output_path = processed_dir / "model_feature_eligibility.csv"
    with duckdb.connect(config={"threads": threads}) as con:
        engineered = con.read_parquet(str(input_path))
        schema = dict(zip(engineered.columns, map(str, engineered.types)))
        inventory = build_feature_inventory(schema)
        coverage = weekly_feature_coverage(engineered, inventory)
        summary = summarize_eligible_coverage(coverage, inventory)
        print("Feature roles:\n" + inventory["role"].value_counts().to_string())
        print("Eligible feature coverage by week:\n" + summary.sort_values("min_coverage_pct").round(2).to_string())
        write_csv_atomic(inventory, output_path)
    eligible_count = int(inventory["role"].eq("eligible").sum())
    print(f"Saved {eligible_count} eligible features to {output_path.resolve()}", flush=True)
    return output_path
