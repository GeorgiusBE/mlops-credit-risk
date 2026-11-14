# Credit risk experiments

The preprocessing notebooks generate `dataset/processed/train_base_engineered.parquet`
and `dataset/processed/model_feature_eligibility.csv`. The baseline script trains
LightGBM with the eligible columns and evaluates on later validation weeks.

Raw inputs and the final training dataset/feature inventory are versioned with
DVC using a local cache. Run `make dvc-status` to inspect them and
`make dvc-snapshot` after changing the data. Commit the updated `.dvc` metadata
to Git to record a version. See [the DVC setup and command guide](docs/data-versioning.md)
for installation, the tracked paths, restoring versions, and adding storage later.

## Project commands with Make

Run `make` or `make help` from the repository root in WSL/Linux to list the
commands. Make uses `.env/bin/python` directly, so activation is optional once
the project environment has been installed. The default command only prints help.

| Command | Action |
| --- | --- |
| `make mlflow` | Start the local server in the foreground; keep this terminal open. |
| `make mlflow-status` | Check server health, version, and the GraphQL API used by the experiment UI. |
| `make baseline` | Train the full eligible-feature baseline. |
| `make ablation-plan` | Inspect the individual-removal feature sets without training or logging. |
| `make ablation` | Train the baseline and seven individual group removals. |
| `make joint-plan` | Inspect the joint-removal plan without training. |
| `make joint` | Run joint removals, save the comparison, and stop for your review. |
| `make confirm-plan` | Inspect your explicit shortlist and temporal validation windows. |
| `make confirm` | Validate your chosen feature sets using a completed joint/review study. |
| `make test` | Run the pytest suite without needing a running MLflow server. |
| `make dvc-status` | Check tracked data against its current DVC snapshot. |
| `make dvc-snapshot` | Cache raw inputs and final training inputs; update DVC metadata. |

Training targets check MLflow first and do not launch a background server.
Use a second terminal for training when the server is running in the first.
The Python entry points below remain available; use `make -n confirm` (or another
target) to print the exact underlying commands without executing them.

You decide what advances at both handoffs: group ablation → joint ablation →
temporal validation. Neither handoff uses an automatic shortlist:

```bash
make mlflow-status
# Review the group-ablation results, then choose at least two groups.
make joint-plan
make joint CANDIDATE_GROUPS="YOUR_FIRST_GROUP YOUR_SECOND_GROUP" RUN_NAME=joint-review-v1
# Review the joint comparison in MLflow before continuing.
make confirm JOINT_STUDY_ID=YOUR_REVIEW_RUN_ID \
  CANDIDATES="YOUR_FIRST_CHOICE YOUR_SECOND_CHOICE" \
  RUN_NAME=temporal-review-v1
```

Replace the uppercase placeholders with your chosen group names, the completed
joint review's parent run ID, and the configuration names you choose from its
table. `CANDIDATE_GROUPS` chooses which groups enter joint testing; `CANDIDATES`
chooses which resulting feature sets enter temporal validation. Choose one or
more reduced sets for temporal validation; the full baseline is always the
control. Both options accept `without_` names: use individual removals such as
`without_deposit_history` in `CANDIDATE_GROUPS`, and individual or combined
removals in `CANDIDATES`. Plain names remain accepted for compatibility.
`make confirm-plan` accepts the same options and reads the recorded
joint plan from MLflow without training or logging new runs.
Keep `group-ablation-first-pass` and its child runs available: joint comparison
reuses that screening study's individual removals. Make does not delete or restore runs.

After running a new screening study, provide its parent run ID explicitly:

```bash
make ablation RUN_NAME=group-ablation-v2
make joint SCREENING_STUDY_ID=your_new_screening_run_id \
  CANDIDATE_GROUPS="YOUR_FIRST_GROUP YOUR_SECOND_GROUP" RUN_NAME=joint-review-v2
```

`SCREENING_STUDY_ID` overrides `screening_parent_run_id` in the in-memory
configuration; the source JSON file stays unchanged. Temporal validation inherits
the screening study and group choices recorded in the joint study, so you do not
need to repeat them. `make joint` trains fresh joint
models by default; it can optionally reuse a compatible `JOINT_STUDY_ID`.
`make confirm` requires a completed joint/review study ID and an explicit human
shortlist. It never starts joint training or chooses candidates automatically.

Other options are `THREADS` (default 2), `CONFIG`, `PROCESSED_DIR`, `TRACKING_URI`,
`EXPERIMENT_NAME`, and `PYTHON`. For example:

```bash
make mlflow PORT=5001
# In another terminal:
make joint PORT=5001 THREADS=2 CANDIDATE_GROUPS="YOUR_FIRST_GROUP YOUR_SECOND_GROUP" RUN_NAME=joint-review-v1
```

`PORT` controls the local server and the default tracking URL. An explicit
`TRACKING_URI` or exported `MLFLOW_TRACKING_URI` takes precedence over that URL.
`EXPERIMENT_NAME` defaults to the exported `MLFLOW_EXPERIMENT_NAME`, or
`credit-risk`. Quote run names or paths containing spaces, for example
`make baseline RUN_NAME="baseline comparison"`. `CONFIG` defaults to
`group_ablation.json` for baseline/ablation and `joint_confirmation.json` for
joint review/confirmation. Use `PYTHON=/path/to/python` for another installed environment.

## Feature-group ablation

The baseline and feature-selection runner share model settings and fixed split
boundaries in `feature_selection/configs/group_ablation.json`. The default study
contains eight configurations: all 20 eligible inputs, followed by seven models
that each exclude one feature group. Training uses weeks 0–63, validation uses
64–77, and test weeks 78–91 remain reserved. If the input extends beyond these
boundaries, explicitly revise the configuration before running another study.

Inspect the exact columns without training or writing MLflow runs:

```bash
python feature_selection/select_features.py --dry-run
```

With the local MLflow server running, execute the first pass:

```bash
python feature_selection/select_features.py --run-name group-ablation-first-pass
```

The runner loads the training/validation inputs once and trains candidates
sequentially using the same parameters, seed, categories, rows, and early
stopping rule. Each model may stop at a different iteration. Metadata and labels
are excluded from the feature groups, and test labels are never loaded. It
rejects overlapping groups, ineligible columns, and eligible features without
a group, so adding features requires an explicit grouping decision.

In the `credit-risk` MLflow experiment, expand the study run to inspect its eight
child runs. Open `ablation_comparison.json` in the study's artifacts for the
comparison table. It includes AUC, Gini stability, changes from the full baseline,
weekly improvement counts, feature counts, best iterations, and child run/model
references. Positive deltas mean that removing the group improved validation
performance. The parent also records the full configuration and exact feature
membership. No permanent exports are written under the processed data directory.

Treat this as screening: confirm promising exclusions together and on additional
chronological validation windows before changing eligibility. Ablation measures
the contribution conditional on the remaining inputs; for example, excluding
raw loan amounts still leaves their information in the affordability ratios.
The script leaves the eligibility inventory unchanged.

Both entry points accept `--config /path/to/config.json`. For later combination
checks, use the joint runner below to generate combinations from groups you
choose. For another validation window, make a configuration copy and revise its
split boundaries while keeping the original final test weeks out of validation.

## Joint removals and temporal confirmation

First review the individual group-ablation results and choose the groups you
want to investigate together. `make joint-plan` lists available groups when no
choice has been supplied. Preview your choices, then produce the joint comparison:

```bash
make joint-plan CANDIDATE_GROUPS="YOUR_FIRST_GROUP YOUR_SECOND_GROUP"
make joint CANDIDATE_GROUPS="YOUR_FIRST_GROUP YOUR_SECOND_GROUP" RUN_NAME=joint-review-v1
```

`candidate_groups` in `feature_selection/configs/joint_confirmation.json` is
empty by default. Supply a space-separated `CANDIDATE_GROUPS` list to Make,
`--candidate-groups` with separate names to the Python script, or edit that JSON
list. Names with the `without_` prefix are accepted here too, so you can copy
individual-removal names from the group-ablation table, for example:

```bash
make joint-plan CANDIDATE_GROUPS="without_deposit_history without_affordability_ratios without_debit_card_history"
make joint CANDIDATE_GROUPS="without_deposit_history without_affordability_ratios without_debit_card_history" RUN_NAME=joint-review-v1
```

These are example choices; no candidate groups are selected by default. Plain
group names still work, including mixed lists of plain and prefixed names.
Both forms resolve to the same group; supplying both for one group is rejected
as a duplicate. Recorded group membership uses plain group names, while model
configuration names retain `without_`. At least two distinct, known groups are
required. The runner generates
every pair and larger removal combination from exactly those groups, fits a
fresh baseline, and combines those results with the chosen groups' existing
individual removals. You do not need to maintain a hard-coded `ablations` map.
Two chosen groups produce one joint removal; three produce three pairs and one
triple removal. More groups increase the number of combinations exponentially.
The joint stage stops for your review and does not start temporal validation.

The configuration also defines the screening study to reuse and three validation
windows. Reuse requires matching dataset and
inventory hashes, parameters, runtime versions, exact feature order, row counts,
and split weeks. If those change, rerun screening and update
`screening_parent_run_id` in the configuration.

Review `combination_comparison.json` and `joint_removal_deltas.png` in the joint
review run; `available_feature_sets.json` lists the exact inputs for every set.
`joint_group_selection.json` records your group choices and whether you supplied
them through the CLI or configuration. Combination names contain the sorted
group names, for example `without_affordability_ratios__and__deposit_history`.
The score ordering is for inspection and does not select a shortlist.
Choose the configurations using performance, weekly behaviour, complexity, and
your domain judgement. Temporal validation compares exactly your choices, plus
the full baseline, on:

| Training weeks | Validation weeks | Final test, unused |
| --- | --- | --- |
| 0–35 | 36–49 | 78–91 |
| 0–49 | 50–63 | 78–91 |
| 0–63 | 64–77 | 78–91 |

Each earlier window learns its categorical vocabularies again from its own
training rows. Future weeks between that window and the final test are recorded
as `unused_future` and are not loaded into its training or validation frames.
Compatible fitted models are reused for the existing 64–77 window. That window
was already used for screening; it is not an independent assessment of the
selected candidates. The earlier windows provide additional stability checks,
while the final test remains reserved until selection and tuning are complete.

You can provide a space-separated `CANDIDATES` list to Make (or `--candidates`
with separate names to the Python script). Alternatively, edit
`confirmation.selected_configurations` in the configuration file. It is empty
by default; there is no automatic top-two fallback or fixed candidate count.
The baseline is added automatically, so omit `all_eligible` from the list.
The `without_` prefix is optional: `deposit_history` and
`without_deposit_history` select the same removal configuration. Similarly,
`debit_card_history__and__deposit_history` selects the recorded
`without_debit_card_history__and__deposit_history` configuration. Exact recorded
names remain supported, including legacy names; comparisons and saved MLflow
records use the recorded names. Supplying both aliases for the same configuration
is rejected as a duplicate.

For example, if your joint study included deposit history and affordability
ratios, and **you choose** those two individual removals for temporal validation:

```bash
make confirm-plan JOINT_STUDY_ID=YOUR_REVIEW_RUN_ID \
  CANDIDATES="without_deposit_history without_affordability_ratios"
make confirm JOINT_STUDY_ID=YOUR_REVIEW_RUN_ID \
  CANDIDATES="without_deposit_history without_affordability_ratios" \
  RUN_NAME=temporal-review-v1
```

The equivalent Python command is:

```bash
python feature_selection/confirm_features.py --stage temporal \
  --joint-study-id YOUR_REVIEW_RUN_ID \
  --candidates without_deposit_history without_affordability_ratios \
  --run-name temporal-review-v1
```

Temporal validation loads the group choices and configuration names from the
reviewed study, including legacy names in existing completed studies. It still
requires your new explicit shortlist; it never inherits or generates a shortlist
from scores. Supplying different groups from the reviewed study is rejected.
For an offline preview, pass `CANDIDATE_GROUPS` to `make confirm-plan` without
`JOINT_STUDY_ID`; actual validation always requires a completed study ID.

The temporal study logs:

- `combination_comparison.json` and `source_runs.json`: all removal patterns for the chosen groups and their source model runs.
- `joint_group_selection.json`: the group choices inherited from the reviewed study.
- `confirmation_candidates.json`: your chosen sets with their exact feature lists, plus the baseline.
- `confirmation_selection.json`: the human shortlist using recorded configuration names, the supplied names before alias resolution, whether they came from the CLI or configuration, and the reviewed study ID.
- `confirmation_by_window.json` and `confirmation_summary.json`: paired changes against each window's own baseline.
- `joint_removal_deltas.png`, `confirmation_deltas.png`, and `confirmation_weekly_gini.png`: readable plots.
- One `summary-...` child run per chosen configuration and baseline, with mean/worst-window AUC and stability deltas for native MLflow charts. These are summary records, not newly fitted models.

After temporal validation, the run is marked `awaiting_human_feature_decision`.
Use the metrics and plots to decide the final feature set. The script does not
choose a winner, create an automatic recommendation, or edit eligibility.
AUC changes and worst-window behaviour remain explicit for your review.

Temporal validation requires `--joint-study-id` (or Make's `JOINT_STUDY_ID`).
It accepts the review parent or a compatible completed fitted joint study.
All commands retain the existing tracking URI, experiment, configuration, and
thread options; no permanent model/report exports are written to processed data.

## Local MLflow

Run these commands from the repository root in WSL/Linux. Activate the existing
Python environment and install the locked dependencies:

```bash
source .env/bin/activate
poetry install --no-root
```

Start the tracking server in a terminal and keep it running:

```bash
python scripts/start_mlflow.py
```

Open <http://127.0.0.1:5000> for the MLflow UI. The launcher uses repository-relative
defaults regardless of the working directory:

| Component | Location |
| --- | --- |
| Tracking API and UI | `http://127.0.0.1:5000` |
| SQLite backend store | `.mlflow/mlflow.db` |
| Artifact store | `.mlflow/artifacts/` |

If port 5000 is already in use, an MLflow instance may already be running. Open
<http://localhost:5000> in your Windows browser; WSL normally forwards its local
server ports to Windows. A second server cannot listen on the same address and
port. The launcher checks for this before starting MLflow.

To restart a foreground instance, press Ctrl+C in its original terminal, then
run the launcher again. The background instance started during setup records its
process ID in `.mlflow/server.pid` and its log in `.mlflow/server.log`. To stop
that instance, read its process ID with `cat .mlflow/server.pid`, then run
`kill <PID>` with that number before restarting the launcher. The SQLite database
and artifact files persist across server restarts.

Artifact uploads and downloads go through the server. New experiments use
`mlflow-artifacts:/...` locations, so the training process does not write to the
artifact directory directly. `.mlflow/` is ignored by Git.

In a second terminal, activate the environment and train the baseline:

```bash
source .env/bin/activate
python feature_selection/train_baseline.py --run-name baseline-all-eligible
```

The `credit-risk` experiment records one run for each training execution:

- Model parameters, feature count, week boundaries, dataset and inventory SHA-256 hashes.
- Validation AUC, average precision, log loss, Gini stability and its components.
- Weekly validation AUC/Gini and case/default counts, with `WEEK_NUM` as the metric step.
- Exact training, validation, and held-out test week lists in `split_weeks.json`.
- The exact feature list, input dtypes, training categorical vocabularies, and runtime package versions.
- A loadable MLflow LightGBM model whose predictions are default probabilities.
- The training script, feature inventory, dependency manifest and lock file.

MLflow is the experiment record: the script prints the run ID and model URI and
logs metrics directly. It does not create duplicate report/model exports in the
processed data directory or a `reports/` artifact folder. The latest test weeks
remain reserved and are not loaded for training or validation.

The logged model expects the selected columns in their saved order, with the
categorical columns prepared using `categorical_levels.json`. Both
`mlflow.lightgbm.load_model(model_uri)` and `mlflow.pyfunc.load_model(model_uri)`
can load it. Automatic model signatures are disabled because MLflow's inferred
string schema rejects these pandas categorical inputs; dtypes are recorded in
`feature_dtypes.json` instead.

Stop the server with Ctrl+C. Restart it with the same command to access previous
runs. To back up the setup, stop the server and copy the whole `.mlflow/` directory,
including both the database and artifacts. A database copy alone does not contain
the fitted models and configuration artifacts.

### Verify the experiment UI API

The experiment page uses GraphQL in addition to the tracking REST API. The
dependency constraints require Graphene 3 and GraphQL-core 3: the older GraphQL 2
stack can let tracking calls succeed while the experiment page receives an HTML
500 error and displays `Unexpected token '<' ... is not valid JSON`.

After installing or updating dependencies, check this endpoint with pytest:

```bash
python -m pytest tests/test_mlflow_ui.py -v
```

Pytest is included in Poetry's `dev` dependency group. With the environment
activated, run `poetry install --with dev --no-root` to install it. Run the full
suite with `make test` or `.env/bin/python -m pytest`. Tests use fresh configuration
fixtures, temporary paths, and parametrized cases; they do not need production
datasets or a running MLflow server. For a focused run, for example:

```bash
.env/bin/python -m pytest tests/test_feature_selection.py -v
.env/bin/python -m pytest -k temporal -v
```

### Configuration and a future remote server

The training script accepts `--tracking-uri`, `--experiment-name`, and `--run-name`.
The first two can also be configured using `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME`. Command-line options take precedence.

For example, use another local port:

```bash
python scripts/start_mlflow.py --port 5001
```

Then, in the training terminal:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
  python feature_selection/train_baseline.py --run-name baseline-port-5001
```

The server launcher also accepts `--data-dir /absolute/path/to/storage`.
Use a consistent location across restarts.

To log future runs to a remote MLflow deployment, set `MLFLOW_TRACKING_URI` to its
endpoint and configure the authentication required by that deployment. The remote
server can use PostgreSQL for metadata and object storage for artifacts while
keeping the training logic the same. Historical local runs require a separate
metadata/artifact migration or can remain in the local archive; changing the URI
does not transfer them automatically.

MLflow references: [Tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/),
[backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/),
[artifact store](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/).
