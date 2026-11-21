# Credit risk experiments

The preprocessing Python pipeline generates `dataset/processed/train_base_engineered.parquet`
and `dataset/processed/model_feature_eligibility.csv`. The baseline script trains
LightGBM with the eligible columns and evaluates on later validation weeks.

For a detailed explanation of preprocessing, feature selection, Optuna tuning,
MLflow provenance, and final refitting, read the
[credit risk modelling workflow reference](docs/credit-risk-modelling-workflow.md).

Raw inputs and the final training dataset/feature inventory are versioned with
DVC using a local cache. Run `make dvc-status` to inspect them and
`make dvc-snapshot` after changing the data. Commit the updated `.dvc` metadata
to Git to record a version. See [the DVC setup and command guide](docs/data-versioning.md)
for installation, the tracked paths, restoring versions, and adding storage later.

## Rebuilding preprocessing outputs

Python scripts in `data_preprocessing` are the supported execution path; the four
notebooks remain unchanged as exploratory references. Run from the repository root:

```bash
make preprocess
# Equivalent command with repository-relative defaults:
.env/bin/python -m data_preprocessing --stage all
# Run one stage using existing upstream outputs:
make preprocess-engineer
# Override directories and execution resources:
make preprocess RAW_DIR=/path/to/raw PROCESSED_DIR=/path/to/processed THREADS=2
```

The CLI supports `--stage join|clean|engineer|validate|all`, `--raw-dir`,
`--processed-dir`, and `--threads` (default: 2). CLI default directories are
resolved from the repository location; explicit relative paths use the working
directory. Each individual stage requires its upstream files and does not run
earlier stages automatically. Python imports do not run preprocessing.

The join stage reads `train/` under the raw directory and writes
`train_base_depth0.parquet`, `train_base_depth1.parquet`, and
`train_base_depth2.parquet`. Cleaning writes `train_base_cleaned.parquet`;
engineering also reads raw histories and writes `train_base_engineered.parquet`;
validation writes `model_feature_eligibility.csv`. Each output is replaced
atomically after its checks pass. A later stage failure can leave earlier stages'
completed outputs in place; the entire pipeline is not one transaction. Use an
alternate processed directory to rebuild without replacing current data.

The inventory preserves all 20 eligible features used for historical comparisons.
The later 17-feature tuning choice remains in the tuning configuration.
Preprocessing does not train models, log MLflow runs, or update DVC snapshots.
After reviewing rebuilt outputs, use `make dvc-snapshot` to record their version.

## Project commands with Make

Run `make` or `make help` from the repository root in WSL/Linux to list the
commands. Make uses `.env/bin/python` directly, so activation is optional once
the project environment has been installed. The default command only prints help.

| Command | Action |
| --- | --- |
| `make preprocess` | Rebuild joins, cleaning, feature engineering, and eligibility in order. |
| `make preprocess-join` | Rebuild the three case-level depth tables from raw training inputs. |
| `make preprocess-clean` | Clean and validate the existing depth-2 table. |
| `make preprocess-engineer` | Add 15 engineered features to the cleaned table. |
| `make preprocess-validate` | Report weekly feature coverage and save the eligibility inventory. |
| `make mlflow` | Start the local server in the foreground; keep this terminal open. |
| `make mlflow-status` | Check server health, version, and the GraphQL API used by the experiment UI. |
| `make baseline` | Train the full eligible-feature baseline. |
| `make ablation-plan` | Inspect the individual-removal feature sets without training or logging. |
| `make ablation` | Train the baseline and seven individual group removals. |
| `make joint-plan` | Inspect the joint-removal plan without training. |
| `make joint` | Run joint removals, save the comparison, and stop for your review. |
| `make confirm-plan` | Inspect your explicit shortlist and temporal validation windows. |
| `make confirm` | Validate your chosen feature sets using a completed joint/review study. |
| `make tune-plan` | Inspect the 17-feature Optuna study, windows, and search space. |
| `make tune` | Run tuning and save the selected fixed-round final refit. |
| `make evaluate-test-plan MODEL_RUN_ID=...` | Check the saved refit and input hashes without reading test rows. |
| `make evaluate-test MODEL_RUN_ID=...` | Evaluate the frozen model on its recorded test weeks and log results. |
| `make serve` | Start the prediction API with the registered MLflow champion. |
| `make ui` | Start the Streamlit form for scoring a credit application. |
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

Other options are `THREADS` (default 2), `CONFIG`, `RAW_DIR`, `PROCESSED_DIR`, `TRACKING_URI`,
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

## Optuna tuning and final refit

The final feature decision is **`without_deposit_history`**, chosen on
15 November 2026: use 17 inputs, excluding `deposit_opening_earliest_days_ago`,
`deposit_opening_latest_days_ago`, and `deposit_opening_span_days`.
The explicit decision, source temporal run, search ranges, and stopping settings
live in `hyperparameter_tuning/configs/lightgbm.json`. The loader projects exactly
these 17 columns from the existing DVC-tracked inputs. The 20-input inventory
and earlier ablation configurations remain the sources for historical comparisons.

Install the updated locked dependencies, inspect the plan, then launch the study
with MLflow running in another terminal:

```bash
source .env/bin/activate
poetry install --no-root
make tune-plan
# First terminal: make mlflow
# Second terminal:
make tune STUDY_NAME=lightgbm-optuna-v1 N_TRIALS=30
```

`N_TRIALS` is the **total attempt budget**, including the first trial with the
current parameter settings, failed/interrupted attempts, and attempts already
recorded when resuming. Each trial trains three separate models using expanding
training windows and validation blocks 36–49, 50–63, and 64–77. AUC early
stopping uses patience 40 and a cap of 2,000 boosting rounds. The control uses
the same enlarged cap as search trials; it is not a reuse of the historical
400-round baseline fit. The seeded TPE sampler maximizes the equally weighted
mean of the three validation Gini stability scores. AUC, average precision,
log loss, worst-window stability, per-window best iterations, duration, and
weekly metrics are also recorded. `reached_round_cap=1` identifies windows
whose best iteration reached the cap; review these before accepting the range.

One MLflow parent in `credit-risk` groups all trial children. Each trial stores
parameters, scores, split weeks, and training-only categorical vocabularies.
Trial models are not saved. After tuning, the selected trial's three best
iterations determine the final round count: take their median, rounding halves
up if an even number of windows is configured. A fresh model learns categories
and fits on all development weeks **0–77**, with that fixed `n_estimators` and
**no early stopping or validation set**. Its `selected-model` child contains the
loadable model, feature order, dtypes, categorical levels, and source trial ID.
Validation scores remain attached to the source trial; they are not reported
as scores for the larger final refit. Weeks **78–91 are never loaded or
evaluated** by the tuning workflow. Final-test evaluation uses the separate
command below.

The parent includes `trial_comparison.json`, `selected_trial.json`,
`selected_model.json`, the full configuration, data/inventory hashes, runtime
versions, and copies/hashes of the runner, shared loader, and dependency files.
All model/report artifacts go through MLflow; they are not written to processed
data. Windows are loaded and released sequentially to bound memory usage.

Optuna's persistent history is in `.optuna/studies.db`, separate from MLflow.
To extend the same study from 30 to 50 total attempts:

```bash
make tune STUDY_NAME=lightgbm-optuna-v1 N_TRIALS=50 RESUME=1
# Optional optimization time budget for an invocation (refit time is additional):
make tune STUDY_NAME=lightgbm-optuna-v2 N_TRIALS=30 TIMEOUT=3600
```

`TIMEOUT` is checked between trials; a running trial finishes all its windows.
The best completed trial is refitted even if the time limit ends optimization
before the attempt budget is exhausted. Reusing a study name requires
`RESUME=1`; resume rejects changes to data, inventory, selected features,
configuration, code, dependencies, runtime, threads, or MLflow destination.
An exclusive local lock prevents simultaneous workers on the same Optuna
database. Interrupted attempts are retained as failed attempts, and resume uses
the same MLflow parent. If the selected trial is unchanged and its final run
finished, resume reuses its saved refit. A new winner gets a new refit; earlier
selected models remain in the experiment history.

Fresh sequential studies are seeded. Resuming retains observations but restarts
the sampler's random generator, so its suggestions need not match an uninterrupted
run exactly. Back up `.optuna/` along with the complete `.mlflow/` directory.
`OPTUNA_DB`, `STUDY_NAME`, `N_TRIALS`, `RESUME`, and `TIMEOUT` supplement the
existing Make options. The equivalent Python entry point is
`python hyperparameter_tuning/tune.py --help`.

These validation blocks have already supported feature selection and now support
tuning; their scores are development evidence, not an independent final-test result.

## Final test evaluation

Once tuning and model selection are finished, evaluate the existing saved
refit. This loads its frozen metadata from MLflow, not the current tuning JSON.
For the selected refit from `lightgbm-optuna-v1`:

```bash
make evaluate-test-plan MODEL_RUN_ID=15b66824a94b48eca0156f0009cbafb0
make evaluate-test MODEL_RUN_ID=15b66824a94b48eca0156f0009cbafb0
```

MLflow must be running. The plan checks model identity, feature/category metadata,
training/test separation, dataset/inventory hashes, and scoring-library versions.
It does not read test rows or create a run. Actual evaluation reads only weeks
78–91, applies the saved 17-feature order, dtypes, and training categorical
levels, and predicts with all 422 saved boosting rounds. Unseen categories
become missing values, with counts recorded. The evaluator never fits a model,
changes its parameters, or chooses a threshold from the test results.

The `selected-model-test` child of `selected-model` records `test_gini_stability`,
`test_roc_auc`, `test_average_precision`, `test_log_loss`, stability components,
weekly metrics, row/default counts, and evaluation provenance. The aggregate
`test_*` metrics also appear on the source `selected-model` run and its MLflow
logged model, so its performance is easy to find. The source refit and study
receive a `test_evaluation_run_id` link and `result_status=test_evaluated`.
The saved training artifacts remain the record of how the original fit was made.

Artifacts include `test_metrics.json`, `test_weekly_metrics.json`,
`unseen_category_counts.json`, `evaluation_provenance.json`, and
`evaluation_summary.json`. A repeated command reuses the completed evaluation
for this model; it does not rescore test rows. An exclusive local lock prevents
two evaluation commands in the checkout from racing. `THREADS`, `PROCESSED_DIR`,
`TRACKING_URI`, `RUN_NAME`, and `PYTHON` are supported Make options.

These are final assessment results for the frozen model. Further parameter or
feature selection based on them would make the test segment development data.

## Prediction API

Start `make mlflow` in one terminal if the tracking server is not already
running. In a second terminal, start the API:

```bash
make serve
```

By default, the API resolves `models:/credit-risk@champion` at startup and loads
that version and the feature metadata from its source run. Restart the API
after moving the alias to pick up the newly promoted version. To serve a
candidate assigned to `challenger`, run `make serve MODEL_ALIAS=challenger`.
`MODEL_NAME` selects the registered model and `TRACKING_URI` points to the
MLflow server. The API listens locally on port `8000`, configurable with
`API_PORT`. [Interactive API docs](http://127.0.0.1:8000/docs)

Send one application with `POST /predict`. Supply the primary applicant's
employment fields and the loan amounts in the same units as the training data.
Include each application field, using `null` for a missing value other than
`decision_date`. Each history list contains event dates for this applicant and
may be omitted when no records are available. Dates use `YYYY-MM-DD`. The
feature calculations use history events strictly before `decision_date` and
treat categories absent from training as missing values.

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{
    "decision_date": "2026-10-01",
    "annuity_amount": 2500,
    "credit_amount": 38000,
    "employment_industry": "IT",
    "income_type": "EMPLOYED",
    "main_occupation_income": 45000,
    "employment_start_date": "2021-04-12",
    "debit_card_opening_dates": ["2020-06-12"],
    "prior_application_dates": ["2024-03-15", "2025-05-10"],
    "bureau_update_dates": ["2026-07-20"]
  }'
```

The response contains one field, `default_probability`, with the model's
estimated probability of default between 0 and 1.

### Streamlit prediction form

With the locked dependencies installed, keep `make mlflow` and `make serve`
running. Start the form in another terminal.

```bash
make ui
```

Enter one application and submit it to see the probability of default as a
percentage. The decision date starts at today. The employment and income
dropdowns use the loaded model's categories, and history dates can be entered
one per line. Set `UI_PORT` to change the Streamlit port or `PREDICTION_API_URL`
to point the form at a different FastAPI base URL. [Open the local form](http://127.0.0.1:8501)

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
