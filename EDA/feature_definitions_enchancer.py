# Enchance the feature definitions CSV with information about which training tables contain each feature.
import re
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from collections import defaultdict

raw_dir = Path("dataset/raw")
if not raw_dir.is_dir():
    raw_dir = Path("..") / raw_dir

training_files = sorted((raw_dir / "train").glob("*.parquet"))
if not training_files:
    raise FileNotFoundError(f"No training Parquet files found in {raw_dir / 'train'}")

training_tables_by_feature = defaultdict(set)
for parquet_file in training_files:
    # Strip the shard suffix while keeping the table depth, e.g. train_static_0_1 -> train_static_0.
    table = re.sub(r"(_[012])_\d+$", r"\1", parquet_file.stem)
    for feature in pq.read_schema(parquet_file).names:
        training_tables_by_feature[feature].add(table)

feature_definitions = pd.read_csv(raw_dir / "feature_definitions.csv", keep_default_na=False)
training_table_labels = {
    feature: "; ".join(sorted(tables))
    for feature, tables in training_tables_by_feature.items()
}
feature_definitions["TrainingTables"] = feature_definitions["Variable"].map(training_table_labels).fillna("")

enhanced_path = raw_dir / "feature_definitions_enhanced.csv"
feature_definitions.to_csv(enhanced_path, index=False)

unmatched = feature_definitions.loc[feature_definitions["TrainingTables"].eq(""), "Variable"]
print(f"Saved {len(feature_definitions)} feature definitions to {enhanced_path.resolve()}")
if not unmatched.empty:
    print("Features absent from the training schemas:", ", ".join(unmatched))