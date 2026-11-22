SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .env/bin/python
PORT ?= 5000
API_PORT ?= 8000
UI_PORT ?= 8501
PREDICTION_API_URL ?= http://127.0.0.1:$(API_PORT)
TRACKING_URI ?= $(or $(MLFLOW_TRACKING_URI),http://127.0.0.1:$(PORT))
EXPERIMENT_NAME ?= $(or $(MLFLOW_EXPERIMENT_NAME),credit-risk)
THREADS ?= 2
RAW_DIR ?= dataset/raw
PROCESSED_DIR ?= dataset/processed
RUN_NAME ?=
SCREENING_STUDY_ID ?=
JOINT_STUDY_ID ?=
CANDIDATES ?=
CANDIDATE_GROUPS ?=
STUDY_NAME ?=
N_TRIALS ?= 30
OPTUNA_DB ?= .optuna/studies.db
TIMEOUT ?=
RESUME ?= 0
MODEL_RUN_ID ?=
MODEL_NAME ?= credit-risk
MODEL_ALIAS ?= champion

baseline ablation ablation-plan: CONFIG ?= feature_selection/configs/group_ablation.json
joint joint-plan confirm confirm-plan: CONFIG ?= feature_selection/configs/joint_confirmation.json
tune tune-plan: CONFIG ?= hyperparameter_tuning/configs/lightgbm.json

# Keep paths, URLs, and run names as single shell arguments, including apostrophes.
shquote = '$(subst ','"'"',$(1))'
PYTHON_CMD = $(call shquote,$(PYTHON))
DVC_CMD = $(PYTHON_CMD) -m dvc
PREPROCESS_ARGS = --raw-dir $(call shquote,$(RAW_DIR)) \
	--processed-dir $(call shquote,$(PROCESSED_DIR)) --threads $(call shquote,$(THREADS))
DVC_DATA = dataset/raw/train dataset/raw/test \
	dataset/raw/feature_definitions.csv dataset/raw/feature_definitions_enhanced.csv \
	dataset/raw/sample_submission.csv \
	dataset/processed/train_base_engineered.parquet \
	dataset/processed/model_feature_eligibility.csv
TRAIN_ARGS = --config $(call shquote,$(CONFIG)) \
	--processed-dir $(call shquote,$(PROCESSED_DIR)) \
	--threads $(call shquote,$(THREADS)) \
	--tracking-uri $(call shquote,$(TRACKING_URI)) \
	--experiment-name $(call shquote,$(EXPERIMENT_NAME)) \
	$(if $(strip $(RUN_NAME)),--run-name $(call shquote,$(RUN_NAME)))
CONFIRM_ARGS = $(if $(strip $(SCREENING_STUDY_ID)),--screening-study-id $(call shquote,$(SCREENING_STUDY_ID))) \
	$(if $(strip $(JOINT_STUDY_ID)),--joint-study-id $(call shquote,$(JOINT_STUDY_ID))) \
	$(if $(strip $(CANDIDATE_GROUPS)),--candidate-groups $(foreach group,$(CANDIDATE_GROUPS),$(call shquote,$(group))))
MANUAL_ARGS = $(if $(strip $(CANDIDATES)),--candidates $(foreach candidate,$(CANDIDATES),$(call shquote,$(candidate))))
TUNE_ARGS = --n-trials $(call shquote,$(N_TRIALS)) --storage-path $(call shquote,$(OPTUNA_DB)) \
	$(if $(strip $(STUDY_NAME)),--study-name $(call shquote,$(STUDY_NAME))) \
	$(if $(strip $(TIMEOUT)),--timeout $(call shquote,$(TIMEOUT))) \
	$(if $(filter 1,$(RESUME)),--resume)
EVAL_ARGS = --processed-dir $(call shquote,$(PROCESSED_DIR)) --threads $(call shquote,$(THREADS)) \
	--tracking-uri $(call shquote,$(TRACKING_URI)) \
	$(if $(strip $(MODEL_RUN_ID)),--model-run-id $(call shquote,$(MODEL_RUN_ID))) \
	$(if $(strip $(RUN_NAME)),--run-name $(call shquote,$(RUN_NAME)))

.PHONY: help check-env mlflow mlflow-status baseline ablation-plan ablation joint-plan joint confirm-plan confirm tune-plan tune evaluate-test-plan evaluate-test serve ui test dvc-status dvc-snapshot
.PHONY: preprocess preprocess-join preprocess-clean preprocess-engineer preprocess-validate
.PHONY: compose-up compose-down

help:
	@printf '%s\n' \
	  'Run from the repository root in WSL/Linux. Python defaults to .env/bin/python.' \
	  '' \
	  'Commands:' \
	  '  make help           Show these instructions (also the default for make).' \
	  '  make preprocess     Rebuild joins, cleaning, features, and eligibility in order.' \
	  '  make preprocess-join      Rebuild the three case-level depth tables.' \
	  '  make preprocess-clean     Clean and validate the existing depth-2 table.' \
	  '  make preprocess-engineer  Add the 15 engineered features to the cleaned table.' \
	  '  make preprocess-validate  Check weekly coverage and save feature eligibility.' \
	  '  make mlflow         Start the local MLflow server in the foreground.' \
	  '  make mlflow-status  Check server health, version, and the experiment UI API.' \
	  '  make baseline       Train the full eligible-feature baseline.' \
	  '  make ablation-plan  Inspect the eight feature sets without training.' \
	  '  make ablation       Run the individual group-ablation study.' \
	  '  make joint-plan     Inspect joint removals without training.' \
	  '  make joint          Run joint removals, save comparisons, and STOP for your review.' \
	  '  make confirm-plan   Inspect the candidates YOU choose and temporal windows.' \
	  '  make confirm        Validate your explicit shortlist using a completed joint study.' \
	  '  make tune-plan      Inspect the 17-feature Optuna study without training or logging.' \
	  '  make tune           Tune on three temporal windows and save the selected final refit.' \
	  '  make evaluate-test-plan  Check a frozen MODEL_RUN_ID without reading test rows.' \
	  '  make evaluate-test  Evaluate that saved model on its reserved test weeks.' \
	  '  make serve         Start the local prediction API with the MLflow champion.' \
	  '  make ui            Start the local Streamlit prediction form.' \
	  '  make compose-up    Start the Docker serving stack and wait for health checks.' \
	  '  make compose-down  Stop and remove the Docker serving containers.' \
	  '  make test           Run the pytest suite; no running server required.' \
	  '  make dvc-status     Check whether tracked data matches its DVC snapshot.' \
	  '  make dvc-snapshot   Cache current raw inputs and final training inputs.' \
	  '' \
	  'Options (NAME=value after the command):' \
	  '  RUN_NAME, THREADS, CONFIG, PROCESSED_DIR, TRACKING_URI, EXPERIMENT_NAME' \
	  '  PORT                Local server port and default tracking URL (5000).' \
	  '  API_PORT            Local prediction API port (8000).' \
	  '  UI_PORT             Local Streamlit port (8501).' \
	  '  PREDICTION_API_URL  FastAPI base URL used by Streamlit (defaults to API_PORT).' \
	  '  PYTHON              Alternate Python executable.' \
	  '  RAW_DIR             Raw preprocessing inputs (dataset/raw).' \
	  '  SCREENING_STUDY_ID   Group-ablation study to reuse; temporal inherits the joint choice.' \
	  '  JOINT_STUDY_ID       Completed joint/review study ID; required for confirm.' \
	  '  CANDIDATE_GROUPS     Groups YOU choose after group ablation for joint testing.' \
	  '                      Accepts without_group_name or plain group_name.' \
	  '                      Choose at least two, or edit candidate_groups in CONFIG.' \
	  '  CANDIDATES          Space-separated configuration names YOU choose.' \
	  '                      The without_ prefix is optional.' \
	  '                      Or edit confirmation.selected_configurations in CONFIG.' \
	  '  STUDY_NAME          Persistent Optuna name; use the same name with RESUME=1.' \
	  '  N_TRIALS            Total trial attempts, including control and earlier attempts (30).' \
	  '  OPTUNA_DB           Local study database (.optuna/studies.db).' \
	  '  TIMEOUT             Optimization seconds for this invocation; refit is additional.' \
	  '  RESUME              Set to 1 to extend a matching existing Optuna study.' \
	  '  MODEL_RUN_ID        Saved final-refit run ID for evaluation.' \
	  '  MODEL_NAME          Registered MLflow model for serving (credit-risk).' \
	  '  MODEL_ALIAS         MLflow model alias for serving (champion).' \
	  '' \
	  'Examples:' \
	  '  make ablation RUN_NAME=group-ablation-v2' \
	  '  make joint CANDIDATE_GROUPS="without_deposit_history without_affordability_ratios" RUN_NAME=joint-review-v1' \
	  '  make confirm JOINT_STUDY_ID=your_review_id CANDIDATES="your_chosen_names"' \
	  '  make -n confirm     Print the commands without executing them.' \
	  '' \
	  'Keep the original screening study and its child runs available for confirm.'

check-env:
	@command -v $(PYTHON_CMD) >/dev/null 2>&1 || { \
	  printf '%s\n' $(call shquote,Python executable not found: $(PYTHON)) \
	    'Install the project environment as described in README.md, or set PYTHON=/path/to/python.' >&2; \
	  exit 1; \
	}

preprocess: check-env
	@$(PYTHON_CMD) -m data_preprocessing $(PREPROCESS_ARGS) --stage all

preprocess-join preprocess-clean preprocess-engineer preprocess-validate: check-env
	@$(PYTHON_CMD) -m data_preprocessing $(PREPROCESS_ARGS) --stage $(patsubst preprocess-%,%,$@)

mlflow: check-env
	@$(PYTHON_CMD) scripts/start_mlflow.py --port $(call shquote,$(PORT))

mlflow-status: check-env
	@$(PYTHON_CMD) scripts/mlflow_status.py --tracking-uri $(call shquote,$(TRACKING_URI))

baseline: mlflow-status
	@$(PYTHON_CMD) feature_selection/train_baseline.py $(TRAIN_ARGS)

ablation-plan: check-env
	@$(PYTHON_CMD) feature_selection/select_features.py $(TRAIN_ARGS) --dry-run

ablation: mlflow-status
	@$(PYTHON_CMD) feature_selection/select_features.py $(TRAIN_ARGS)

joint-plan: check-env
	@$(PYTHON_CMD) feature_selection/confirm_features.py $(TRAIN_ARGS) $(CONFIRM_ARGS) --stage joint --dry-run

joint: mlflow-status
	@$(PYTHON_CMD) feature_selection/confirm_features.py $(TRAIN_ARGS) $(CONFIRM_ARGS) --stage joint

confirm-plan: check-env
	@$(PYTHON_CMD) feature_selection/confirm_features.py $(TRAIN_ARGS) $(CONFIRM_ARGS) $(MANUAL_ARGS) --stage temporal --dry-run

confirm: mlflow-status
	@$(PYTHON_CMD) feature_selection/confirm_features.py $(TRAIN_ARGS) $(CONFIRM_ARGS) $(MANUAL_ARGS) --stage temporal

tune-plan: check-env
	@$(PYTHON_CMD) hyperparameter_tuning/tune.py $(TRAIN_ARGS) $(TUNE_ARGS) --dry-run

tune: mlflow-status
	@$(PYTHON_CMD) hyperparameter_tuning/tune.py $(TRAIN_ARGS) $(TUNE_ARGS)

evaluate-test-plan: mlflow-status
	@$(PYTHON_CMD) model_evaluation/evaluate_test.py $(EVAL_ARGS) --dry-run

evaluate-test: mlflow-status
	@$(PYTHON_CMD) model_evaluation/evaluate_test.py $(EVAL_ARGS)

serve: check-env
	@MLFLOW_TRACKING_URI=$(call shquote,$(TRACKING_URI)) \
	  MODEL_NAME=$(call shquote,$(MODEL_NAME)) MODEL_ALIAS=$(call shquote,$(MODEL_ALIAS)) \
	  $(PYTHON_CMD) -m uvicorn prediction_api:app --host 127.0.0.1 --port $(call shquote,$(API_PORT))

ui: check-env
	@PREDICTION_API_URL=$(call shquote,$(PREDICTION_API_URL)) \
	  $(PYTHON_CMD) -m streamlit run streamlit_app.py --server.headless=true \
	  --server.address=127.0.0.1 --server.port=$(call shquote,$(UI_PORT))

compose-up:
	@docker compose up --wait

compose-down:
	@docker compose down

test: check-env
	@$(PYTHON_CMD) -m pytest

dvc-status: check-env
	@$(DVC_CMD) status

dvc-snapshot: check-env
	@$(DVC_CMD) add $(DVC_DATA)
