"""Score a synthetic application with the built image and a temporary MLflow registry."""

from importlib.metadata import version
from math import isclose
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import requests

from prediction_api import INCOME_TYPE_FEATURE


MODEL_NAME = "credit-risk-smoke"
MODEL_ALIAS = "champion"
INCOME_TYPES = ["EMPLOYED", "RETIRED"]
APPLICATION = {
    "decision_date": "2026-10-01",
    "credit_amount": 20000,
    "annuity_amount": None,
    "employment_industry": None,
    "income_type": "EMPLOYED",
    "main_occupation_income": 100000,
    "employment_start_date": None,
}


def seed_model(tracking_url):
    """Register a model with the same input artifacts used by the prediction API."""
    frame = pd.DataFrame({
        "credit_amount_to_reported_income": [0.2, 0.3, 0.7, 0.8] * 2,
        INCOME_TYPE_FEATURE: pd.Categorical(
            ["EMPLOYED"] * 4 + ["RETIRED"] * 4, categories=INCOME_TYPES,
        ),
    })
    model = lgb.LGBMClassifier(
        n_estimators=5, num_leaves=2, min_child_samples=1, min_data_in_bin=1,
        n_jobs=1, random_state=0, verbosity=-1,
    ).fit(frame, [0, 0, 1, 1] * 2)
    # The application above independently maps to the first training row.
    expected = float(model.booster_.predict(frame.iloc[[0]], num_threads=1)[0])
    runtime_versions = {
        name: version(name) for name in ("lightgbm", "pandas", "numpy", "scipy", "scikit-learn")
    }

    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_registry_uri(tracking_url)
    mlflow.set_experiment(MODEL_NAME)
    features = list(frame.columns)
    with mlflow.start_run():
        mlflow.log_dict({"features": features, "runtime_versions": runtime_versions}, "provenance.json")
        mlflow.log_dict({"features": features, "categorical_features": [INCOME_TYPE_FEATURE]}, "features.json")
        mlflow.log_dict(frame.dtypes.astype(str).to_dict(), "feature_dtypes.json")
        mlflow.log_dict({INCOME_TYPE_FEATURE: INCOME_TYPES}, "categorical_levels.json")
        logged_model = mlflow.lightgbm.log_model(
            model.booster_, name="model", signature=False,
            pip_requirements=[f"lightgbm=={runtime_versions['lightgbm']}"],
        )
    registered = mlflow.register_model(logged_model.model_uri, MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, registered.version)
    return f"models:/{MODEL_NAME}/{registered.version}", expected


def main():
    repo = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="credit-risk-smoke-") as data_dir:
        command = ["docker", "compose", "--file", "compose.yaml", "--project-name", Path(data_dir).name]
        env = dict(
            os.environ, MLFLOW_DATA_DIR=data_dir, MLFLOW_UID=str(os.getuid()),
            MLFLOW_GID=str(os.getgid()), PORT="0", API_PORT="0", UI_PORT="0",
            MODEL_NAME=MODEL_NAME, MODEL_ALIAS=MODEL_ALIAS,
        )

        def compose(*args, capture_output=False):
            return subprocess.run(
                [*command, *args], cwd=repo, env=env, check=True,
                text=True, capture_output=capture_output,
            ).stdout

        start = ("up", "--no-build", "--wait", "--wait-timeout", "180")
        try:
            compose(*start, "mlflow")
            registry = compose("port", "mlflow", "5000", capture_output=True).strip()
            model_uri, expected = seed_model(f"http://{registry}")

            compose(*start)
            logs = compose("logs", "--no-color", "api", capture_output=True)
            if f"Loaded {MODEL_NAME}@{MODEL_ALIAS} from {model_uri}" not in logs:
                raise RuntimeError(f"FastAPI did not log loading {model_uri}.")

            api = compose("port", "api", "8000", capture_output=True).strip()
            options = requests.get(f"http://{api}/predict/options", timeout=10)
            options.raise_for_status()
            if options.json()["income_types"] != INCOME_TYPES:
                raise RuntimeError(f"Unexpected model categories: {options.text}")

            prediction = requests.post(f"http://{api}/predict", json=APPLICATION, timeout=10)
            prediction.raise_for_status()
            probability = prediction.json()["default_probability"]
            if not isclose(probability, expected, rel_tol=0, abs_tol=1e-9):
                raise RuntimeError(f"Expected default probability {expected:.12f}, got {probability:.12f}.")
            print(f"Compose scored the synthetic application with {model_uri}: {probability:.12f}")
        except Exception:
            subprocess.run([*command, "logs", "--no-color", "--tail", "80"], cwd=repo, env=env, check=False)
            raise
        finally:
            compose("down")


if __name__ == "__main__":
    main()
