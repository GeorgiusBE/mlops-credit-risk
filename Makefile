SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .env/bin/python
PORT ?= 5000
TRACKING_URI ?= $(or $(MLFLOW_TRACKING_URI),http://127.0.0.1:$(PORT))
EXPERIMENT_NAME ?= $(or $(MLFLOW_EXPERIMENT_NAME),credit-risk)
THREADS ?= 2
PROCESSED_DIR ?= dataset/processed
RUN_NAME ?=
SCREENING_STUDY_ID ?=
JOINT_STUDY_ID ?=
CANDIDATES ?=
CANDIDATE_GROUPS ?=

baseline ablation ablation-plan: CONFIG ?= feature_selection/configs/group_ablation.json
joint joint-plan confirm confirm-plan: CONFIG ?= feature_selection/configs/joint_confirmation.json

# Keep paths, URLs, and run names as single shell arguments, including apostrophes.
shquote = '$(subst ','"'"',$(1))'
PYTHON_CMD = $(call shquote,$(PYTHON))
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

.PHONY: help check-env mlflow mlflow-status baseline ablation-plan ablation joint-plan joint confirm-plan confirm test

help:
	@printf '%s\n' \
	  'Run from the repository root in WSL/Linux. Python defaults to .env/bin/python.' \
	  '' \
	  'Commands:' \
	  '  make help           Show these instructions (also the default for make).' \
	  '  make mlflow         Start the local MLflow server in the foreground.' \
	  '  make mlflow-status  Check server health, version, and the experiment UI API.' \
	  '  make baseline       Train the full eligible-feature baseline.' \
	  '  make ablation-plan  Inspect the eight feature sets without training.' \
	  '  make ablation       Run the individual group-ablation study.' \
	  '  make joint-plan     Inspect joint removals without training.' \
	  '  make joint          Run joint removals, save comparisons, and STOP for your review.' \
	  '  make confirm-plan   Inspect the candidates YOU choose and temporal windows.' \
	  '  make confirm        Validate your explicit shortlist using a completed joint study.' \
	  '  make test           Run the test suite; no running server required.' \
	  '' \
	  'Options (NAME=value after the command):' \
	  '  RUN_NAME, THREADS, CONFIG, PROCESSED_DIR, TRACKING_URI, EXPERIMENT_NAME' \
	  '  PORT                Local server port and default tracking URL (5000).' \
	  '  PYTHON              Alternate Python executable.' \
	  '  SCREENING_STUDY_ID   Group-ablation study to reuse; temporal inherits the joint choice.' \
	  '  JOINT_STUDY_ID       Completed joint/review study ID; required for confirm.' \
	  '  CANDIDATE_GROUPS     Groups YOU choose after group ablation for joint testing.' \
	  '                      Choose at least two, or edit candidate_groups in CONFIG.' \
	  '  CANDIDATES          Space-separated configuration names YOU choose.' \
	  '                      Or edit confirmation.selected_configurations in CONFIG.' \
	  '' \
	  'Examples:' \
	  '  make ablation RUN_NAME=group-ablation-v2' \
	  '  make joint CANDIDATE_GROUPS="your_chosen_groups" RUN_NAME=joint-review-v1' \
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

test: check-env
	@$(PYTHON_CMD) -m unittest discover -s tests -v
