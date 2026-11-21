# Dataset versioning with DVC

DVC stores data contents in `.dvc/cache/`; Git records the small `.dvc` metadata
files identifying those contents. This project currently has no DVC remote.
Keep the cache to retain local versions. Cloning the Git repository alone will
not retrieve the datasets. Deleting the cache or running garbage collection can
make historical versions unavailable if no other copy exists.

## What is tracked

| Data | DVC metadata |
| --- | --- |
| Raw training tables, including all files under `dataset/raw/train/` | `dataset/raw/train.dvc` |
| Raw sample test tables under `dataset/raw/test/` | `dataset/raw/test.dvc` |
| Original feature definitions | `dataset/raw/feature_definitions.csv.dvc` |
| Enhanced feature definitions used by preprocessing | `dataset/raw/feature_definitions_enhanced.csv.dvc` |
| Sample submission | `dataset/raw/sample_submission.csv.dvc` |
| Final engineered training table | `dataset/processed/train_base_engineered.parquet.dvc` |
| Matching feature eligibility inventory | `dataset/processed/model_feature_eligibility.csv.dvc` |

The depth-join and cleaned intermediate Parquet files remain outside DVC.
MLflow continues to store experiment metrics and models separately in `.mlflow/`.
The diagram is also outside DVC. No DVC preprocessing or training stages
have been defined: taking a snapshot records existing files without running
preprocessing or training. Use `make preprocess` to rebuild with the Python
pipeline. Snapshot the engineered table and eligibility inventory
after both have been generated successfully.

## Initial setup commands

These commands were run from the repository root during setup. Initialization is
already complete; do not run `dvc init` again in this checkout.

```bash
# Explicitly select the same environment used by Make and training.
VIRTUAL_ENV="$PWD/.env" poetry add --group dev 'dvc>=3.67,<4' --lock
VIRTUAL_ENV="$PWD/.env" poetry install --only dev --no-root
.env/bin/python -m dvc init
.env/bin/python -m dvc config core.analytics false
.env/bin/python -m dvc config cache.type reflink,copy
make dvc-snapshot
```

DVC is a development dependency in `pyproject.toml`, pinned with its dependencies
in `poetry.lock`. For a fresh environment, install main and development
dependencies together with `poetry install --with dev --no-root` after activating
that environment. The `--only dev` command above added DVC to the existing
environment where training dependencies were already installed.

`cache.type = reflink,copy` requests filesystem copy-on-write links when supported
and ordinary copies otherwise. Workspace files can be edited independently of
cached versions. The cache, temporary files, and local configuration are ignored
by Git. Root `.gitignore` rules allow DVC metadata while keeping dataset contents
and preprocessing intermediates out of Git.

## Check or save a version

```bash
make dvc-status
make dvc-snapshot
```

The snapshot target runs exactly:

```bash
.env/bin/python -m dvc add \
  dataset/raw/train dataset/raw/test \
  dataset/raw/feature_definitions.csv \
  dataset/raw/feature_definitions_enhanced.csv \
  dataset/raw/sample_submission.csv \
  dataset/processed/train_base_engineered.parquet \
  dataset/processed/model_feature_eligibility.csv
```

It caches changed contents and updates metadata. It does not commit to Git.
Review and commit the metadata and related code together to record a named point
in history. For the initial setup, review these exact paths:

```bash
git diff -- .gitignore Makefile README.md pyproject.toml poetry.lock
git status --short
git add .dvc/.gitignore .dvc/config .dvcignore .gitignore \
  dataset/raw/*.dvc dataset/processed/*.dvc \
  pyproject.toml poetry.lock Makefile README.md docs/data-versioning.md
git diff --cached
git commit -m "Set up local DVC dataset versioning"
```

The setup left pre-existing workspace edits in place. In particular, Makefile
and README changes may include earlier work; review those before committing.
DVC initialization automatically stages its initial configuration files.

For later dataset updates, rerun `make dvc-snapshot`, then commit the changed
`.dvc` files with the preprocessing/configuration changes that produced the data.
Dataset edits are not recorded automatically. Adding new files inside the tracked
raw train/test directories is picked up by the snapshot target; a new dataset
outside those paths needs its own `dvc add` command and suitable Git ignore rules.

## Restore a recorded version

First commit or otherwise preserve current code and dataset changes. Replace
`<commit-or-tag>` below with a Git revision containing the desired DVC metadata:

```bash
git switch --detach <commit-or-tag>
.env/bin/python -m dvc checkout
```

Git selects the version's metadata; DVC restores its matching contents from the
local cache. `dvc checkout` updates workspace datasets, so save valuable unrecorded
data changes before running it. To return to the branch you were on:

```bash
git switch -
.env/bin/python -m dvc checkout
```

Only snapshots whose metadata was committed to Git can be selected this way.
Existing historical MLflow runs have not been retroactively linked to DVC;
the initial snapshot covers the data present at setup time.

## Add backup or shared storage later

A mounted external drive can be used without a cloud account. Replace the example
path with a real persistent backup location on that drive:

```bash
mkdir -p /mnt/backup/credit-risk-dvc
.env/bin/python -m dvc remote add --local -d backup /mnt/backup/credit-risk-dvc
.env/bin/python -m dvc push
```

The machine-specific remote is kept in Git-ignored `.dvc/config.local`. `dvc push`
uploads the currently selected data; to back up versions referenced by all local
branches and tags, use `dvc push --all-branches --all-tags`. For restoring a Git
revision whose data is missing locally, select its metadata and run `dvc pull`.
Back up `.mlflow/` separately using the procedure in the main README.

Official references: [DVC add](https://doc.dvc.org/command-reference/add),
[checkout](https://doc.dvc.org/command-reference/checkout),
[remote add](https://doc.dvc.org/command-reference/remote/add).

## Setup verification

The initial snapshot contains 73 files across seven tracked outputs. Content
hashes taken before and after setup matched for all 73 files. The feature
inventory was temporarily removed with a backup held outside the repository,
restored with `dvc checkout dataset/processed/model_feature_eligibility.csv.dvc`,
and compared byte for byte with the backup. `make dvc-status` reports that the
data is up to date, and DVC discovers all seven metadata files automatically.
