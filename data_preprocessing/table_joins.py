"""Join depth-0 tables and case-level summaries of depth-1/depth-2 histories."""

from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from .common import (
    quote_identifier as quote,
    read_training_history,
    require_file,
    validate_case_grain,
    write_parquet_atomic,
)

DEPTH1_SOURCES = ("debitcard", "deposit", "other", "tax_registry_a", "tax_registry_b", "tax_registry_c")
PAIRED_SOURCES = ("applprev", "credit_bureau_a", "credit_bureau_b", "person")


def join_depth0(train_dir, output_path, threads):
    base_path = require_file(train_dir / "train_base.parquet", "Restore the raw training inputs first.")
    with duckdb.connect(config={"threads": threads, "preserve_insertion_order": False}) as con:
        base = con.read_parquet(str(base_path))
        static_tables = [
            (source, read_training_history(con, train_dir, source, 0))
            for source in ("static", "static_cb")
        ]
        expected_rows = validate_case_grain(base, name="train_base")
        joined = base
        for source, table in static_tables:
            validate_case_grain(table, name=f"train_{source}_0", allow_empty=True)
            features = [
                duckdb.ColumnExpression(column).alias(f"{source}_0_{column}")
                for column in table.columns if column != "case_id"
            ]
            prefixed = table.select(duckdb.ColumnExpression("case_id"), *features)
            joined = joined.join(prefixed, "case_id", how="left")
        validate_case_grain(joined, expected_rows=expected_rows)
        write_parquet_atomic(joined, output_path)
    print(f"Saved {output_path.resolve()}", flush=True)


def join_depth1(train_dir, input_path, output_path, threads):
    with duckdb.connect(config={"threads": threads, "preserve_insertion_order": False}) as con:
        joined = con.read_parquet(str(input_path))
        expected_rows = validate_case_grain(joined)
        count_columns = []
        for source in DEPTH1_SOURCES:
            table = read_training_history(con, train_dir, source, 1)
            count_column = f"{source}_record_count"
            count_columns.append(count_column)
            aggregations = ["case_id", f"count(*) AS {count_column}"]
            amount_stats = ("avg",) if source == "other" else ("avg", "max")
            for column in table.columns:
                feature = f"{source}_{column}"
                if column.endswith("A"):
                    aggregations.extend(f"{stat}({quote(column)}) AS {quote(f'{feature}_{stat}')}" for stat in amount_stats)
                elif column.endswith("D"):
                    aggregations.extend(
                        f"{stat}(CAST({quote(column)} AS DATE)) AS {quote(f'{feature}_{stat}')}"
                        for stat in ("min", "max")
                    )
                elif column.endswith("M"):
                    aggregations.append(f"count(DISTINCT {quote(column)}) AS {quote(f'{feature}_distinct_count')}")
            features = table.aggregate(", ".join(aggregations), "case_id")
            joined = joined.join(features, "case_id", how="left")
        replacements = ", ".join(f"coalesce({name}, 0) AS {name}" for name in count_columns)
        joined = joined.project(f"* REPLACE ({replacements})")
        validate_case_grain(joined, expected_rows=expected_rows)
        write_parquet_atomic(joined, output_path)
    print(f"Saved {output_path.resolve()}", flush=True)


def history_statistics(table, prefix):
    """Yield the notebook's per-parent statistics and their case-level rollups."""
    for column, dtype in zip(table.columns, table.types):
        if column in {"case_id", "num_group1", "num_group2"}:
            continue
        if column.endswith("D"):
            value, stats = f"CAST({quote(column)} AS DATE)", ("min", "max")
        elif str(dtype) == "VARCHAR":
            yield f"{prefix}_{column}_distinct_count", f"count(DISTINCT {quote(column)})", "max"
            continue
        elif str(dtype) == "BOOLEAN":
            yield f"{prefix}_{column}_true_share", f"avg(CAST({quote(column)} AS INTEGER))", "avg"
            continue
        else:
            value = quote(column)
            stats = ("min", "max") if column.endswith("T") else ("avg", "max")
        for stat in stats:
            yield f"{prefix}_{column}_{stat}", f"{stat}({value})", stat


def summarize_history_pair(con, train_dir, source, work_dir):
    parents = read_training_history(con, train_dir, source, 1)
    details = read_training_history(con, train_dir, source, 2)
    parent_count = f"{source}_1_record_count"
    detail_count = f"{source}_2_record_count"
    parents_with_details = f"{source}_2_parent_count"

    parent_stats = ["case_id", f"count(*) AS {parent_count}"]
    if source == "person":
        parent_stats.extend(
            f"max({quote(column)}) FILTER (WHERE num_group1 = 0) AS {quote(f'person_1_applicant_{column}')}"
            for column in parents.columns if column not in {"case_id", "num_group1"}
        )
    else:
        parent_stats.extend(
            f"{expression} AS {quote(name)}" for name, expression, _ in history_statistics(parents, f"{source}_1")
        )
    parent_path = work_dir / f"{source}_1_summary.parquet"
    parents.aggregate(", ".join(parent_stats), "case_id").write_parquet(str(parent_path))

    parent_key = "case_id, num_group1"
    detail_stats = [parent_key, "count(*) AS rows_per_parent"]
    case_stats = [
        "case_id",
        f"CAST(sum(rows_per_parent) AS BIGINT) AS {detail_count}",
        f"count(*) AS {parents_with_details}",
        f"max(rows_per_parent) AS {source}_2_max_records_per_parent",
    ]
    for name, expression, rollup in history_statistics(details, f"{source}_2"):
        detail_stats.append(f"{expression} AS {quote(name)}")
        if name.endswith("_distinct_count"):
            output_name = f"{name}_max_per_parent"
        elif rollup == "avg":
            output_name = f"{name}_across_parents"
        else:
            output_name = name
        case_stats.append(f"{rollup}({quote(name)}) AS {quote(output_name)}")

    details_by_parent = details.aggregate(", ".join(detail_stats), parent_key)
    matched_details = details_by_parent.join(parents.project(parent_key), parent_key, how="semi")
    detail_path = work_dir / f"{source}_2_summary.parquet"
    matched_details.aggregate(", ".join(case_stats), "case_id").write_parquet(str(detail_path))
    return (parent_path, detail_path), (parent_count, detail_count, parents_with_details)


def join_depth2(train_dir, input_path, output_path, threads):
    # Materialized summaries and hash batches bound the large bureau joins.
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        work_dir = Path(temp_dir)
        with duckdb.connect(config={
            "threads": threads,
            "preserve_insertion_order": False,
            "memory_limit": "2GB",
            "temp_directory": str(work_dir / "spill"),
        }) as con:
            expected_rows = validate_case_grain(con.read_parquet(str(input_path)))
            summaries, count_columns = [], []
            for source in PAIRED_SOURCES:
                paths, counts = summarize_history_pair(con, train_dir, source, work_dir)
                summaries.extend(paths)
                count_columns.extend(counts)
                print(f"Summarized {source}", flush=True)
            replacements = ", ".join(f"coalesce({name}, 0) AS {name}" for name in count_columns)
            batch_paths = []
            batch_count = 8
            for batch in range(batch_count):
                case_filter = f"hash(case_id) % {batch_count} = {batch}"
                joined = con.read_parquet(str(input_path)).filter(case_filter)
                for path in summaries:
                    summary = con.read_parquet(str(path)).filter(case_filter)
                    joined = joined.join(summary, "case_id", how="left")
                joined = joined.project(f"* REPLACE ({replacements})")
                batch_path = work_dir / f"batch_{batch}.parquet"
                joined.write_parquet(str(batch_path), row_group_size=32768)
                batch_paths.append(str(batch_path))
                print(f"Joined case batch {batch + 1}/{batch_count}", flush=True)
            joined = con.read_parquet(batch_paths)
            validate_case_grain(joined, expected_rows=expected_rows)
            write_parquet_atomic(joined, output_path, row_group_size=32768)
    print(f"Saved {output_path.resolve()}", flush=True)


def run(raw_dir, processed_dir, threads=2):
    """Rebuild the three case-level depth tables and return their paths."""
    train_dir = Path(raw_dir) / "train"
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(processed_dir / f"train_base_depth{depth}.parquet" for depth in range(3))
    join_depth0(train_dir, paths[0], threads)
    join_depth1(train_dir, paths[0], paths[1], threads)
    join_depth2(train_dir, paths[1], paths[2], threads)
    return paths
