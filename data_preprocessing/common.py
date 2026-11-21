"""Paths, validation, and atomic output helpers shared by preprocessing stages."""

from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def quote_identifier(name):
    """Quote a DuckDB column identifier, including embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


def require_file(path, instruction):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.resolve()}. {instruction}")
    return path


def training_paths(train_dir, source, depth):
    """Use a single history file when present, otherwise all sorted row shards."""
    train_dir = Path(train_dir)
    stem = f"train_{source}_{depth}"
    single_path = train_dir / f"{stem}.parquet"
    paths = [single_path] if single_path.is_file() else sorted(train_dir.glob(f"{stem}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found for {stem} in {train_dir.resolve()}")
    return paths


def read_training_history(con, train_dir, source, depth):
    return con.read_parquet([str(path) for path in training_paths(train_dir, source, depth)])


def validate_case_grain(relation, *, expected_rows=None, name="Training table", allow_empty=False):
    """Reject empty tables, null/duplicate case IDs, and changed case counts."""
    [(rows, unique_ids)] = relation.aggregate("count(*), count(DISTINCT case_id)").fetchall()
    return check_case_counts(rows, unique_ids, expected_rows=expected_rows, name=name, allow_empty=allow_empty)


def check_case_counts(rows, unique_ids, *, expected_rows=None, name="Training table", allow_empty=False):
    if rows == 0 and not allow_empty:
        raise ValueError(f"{name} is empty.")
    if rows != unique_ids:
        raise ValueError(f"{name} must have one row per non-null case_id.")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{name} must preserve {expected_rows:,} input rows; found {rows:,}.")
    return rows


def validate_training_table(relation, *, expected_rows=None):
    """Validate case grain, binary labels, and required time metadata."""
    [counts] = relation.aggregate("""
        count(*), count(DISTINCT case_id),
        count(*) FILTER (WHERE target IS NULL OR target NOT IN (0, 1)),
        count(*) FILTER (WHERE WEEK_NUM IS NULL),
        count(*) FILTER (WHERE date_decision IS NULL)
    """).fetchall()
    return check_training_counts(*counts, expected_rows=expected_rows)


def check_training_counts(rows, unique_ids, invalid_targets, missing_weeks, missing_dates, *, expected_rows=None):
    """Check combined aggregate results without rescanning a large lazy relation."""
    check_case_counts(rows, unique_ids, expected_rows=expected_rows)
    for count, description in (
        (invalid_targets, "missing or non-binary targets"),
        (missing_weeks, "rows with a missing WEEK_NUM"),
        (missing_dates, "rows with a missing date_decision"),
    ):
        if count:
            raise ValueError(f"Found {count:,} {description}.")
    return rows


def write_parquet_atomic(relation, output_path, **parquet_options):
    """Replace an output only after DuckDB finishes writing its temporary file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        relation.write_parquet(str(temp_path), **parquet_options)
        temp_path.replace(output_path)


def write_csv_atomic(frame, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        frame.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
