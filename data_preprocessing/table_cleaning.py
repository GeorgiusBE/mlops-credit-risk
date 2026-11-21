"""Standardize dates, infinities, and blank strings in the joined training table."""

from pathlib import Path

import duckdb

from .common import quote_identifier as quote, require_file, validate_training_table, write_parquet_atomic


def invalid_value_counts(cases, kind):
    """Count infinities or blank text, returning an empty dict for absent types."""
    expressions, names = [], []
    for name, dtype in zip(cases.columns, cases.types):
        column = quote(name)
        if kind == "infinity" and str(dtype) in {"FLOAT", "DOUBLE"}:
            condition = f"isinf({column})"
        elif kind == "blank" and str(dtype) == "VARCHAR":
            condition = rf"regexp_full_match({column}, '\s*')"
        else:
            continue
        names.append(name)
        expressions.append(f"count(*) FILTER (WHERE {condition})")
    if not expressions:
        return {}
    [counts] = cases.aggregate(", ".join(expressions)).fetchall()
    return dict(zip(names, counts))


def clean_table(cases):
    """Return a lazy cleaned relation with the original row and column order."""
    date_casts = [
        f"CAST({quote(name)} AS DATE) AS {quote(name)}"
        for name, dtype in zip(cases.columns, cases.types)
        if str(dtype) == "VARCHAR" and (name.endswith("D") or name == "date_decision")
    ]
    cleaned = cases.project(f"* REPLACE ({', '.join(date_casts)})") if date_casts else cases

    infinities = invalid_value_counts(cleaned, "infinity")
    replacements = [
        f"CASE WHEN isinf({quote(name)}) THEN NULL ELSE {quote(name)} END AS {quote(name)}"
        for name, count in infinities.items() if count
    ]
    if replacements:
        cleaned = cleaned.project(f"* REPLACE ({', '.join(replacements)})")

    blanks = invalid_value_counts(cleaned, "blank")
    replacements = [
        rf"CASE WHEN regexp_full_match({quote(name)}, '\s*') THEN NULL ELSE {quote(name)} END AS {quote(name)}"
        for name, count in blanks.items() if count
    ]
    if replacements:
        cleaned = cleaned.project(f"* REPLACE ({', '.join(replacements)})")
    print(f"Replaced {sum(infinities.values()):,} infinities and {sum(blanks.values()):,} blank strings; "
          f"cast {len(date_casts)} date columns.", flush=True)
    return cleaned


def run(raw_dir, processed_dir, threads=2):
    """Validate and save train_base_cleaned.parquet; raw_dir is unused."""
    processed_dir = Path(processed_dir)
    input_path = require_file(processed_dir / "train_base_depth2.parquet", "Run the join stage: make preprocess-join.")
    output_path = processed_dir / "train_base_cleaned.parquet"
    with duckdb.connect(config={"threads": threads, "preserve_insertion_order": False}) as con:
        null_summary = con.execute("""
            SELECT path_in_schema AS feature,
                   CAST(SUM(stats_null_count) AS BIGINT) AS null_count,
                   ROUND(100.0 * SUM(stats_null_count) / SUM(row_group_num_rows), 5) AS null_pct
            FROM parquet_metadata(?)
            GROUP BY path_in_schema ORDER BY null_count DESC, feature
        """, [str(input_path)]).df()
        print("Input null summary (first 10 columns):\n" + null_summary.head(10).to_string(index=False))
        cleaned = clean_table(con.read_parquet(str(input_path)))
        rows = validate_training_table(cleaned)
        for kind in ("infinity", "blank"):
            remaining = [name for name, count in invalid_value_counts(cleaned, kind).items() if count]
            if remaining:
                raise ValueError(f"Unclean {kind} values remain in: {', '.join(remaining)}")
        write_parquet_atomic(cleaned, output_path, row_group_size=32768)
    print(f"Validated and saved {rows:,} cases to {output_path.resolve()}", flush=True)
    return output_path
