"""Run the four preprocessing stages in order, or rebuild one selected stage."""

import argparse
from pathlib import Path

import duckdb

from . import feature_engineering, model_input_validation, table_cleaning, table_joins
from .common import PROJECT_ROOT

STAGES = {
    "join": table_joins.run,
    "clean": table_cleaning.run,
    "engineer": feature_engineering.run,
    "validate": model_input_validation.run,
}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "dataset" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "dataset" / "processed")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    try:
        for name, run in stages.items():
            print(f"Starting preprocessing stage: {name}", flush=True)
            run(args.raw_dir, args.processed_dir, args.threads)
    except (FileNotFoundError, ValueError, duckdb.Error) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")


if __name__ == "__main__":
    main()
