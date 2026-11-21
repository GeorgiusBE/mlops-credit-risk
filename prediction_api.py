"""Serve default probabilities for one raw credit application at a time."""

from contextlib import asynccontextmanager
from datetime import date
import logging
from math import isinf
import os
from threading import Lock
from typing import Annotated

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import mlflow
import pyarrow as pa
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from data_preprocessing.feature_engineering import HISTORY_DATE_SOURCES, add_history_features
from data_preprocessing.table_cleaning import clean_table
from model_evaluation.evaluate_test import (
    check_scoring_runtime,
    load_frozen_booster,
    predict_probabilities,
    prepare_model_inputs,
    read_model_input_metadata,
)


def parse_date(value):
    if type(value) is date:
        return value
    if isinstance(value, str) and len(value) == 10:
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError("Use a date in YYYY-MM-DD format.") from None
        if parsed.isoformat() == value:
            return parsed
    raise ValueError("Use a date in YYYY-MM-DD format.")


CalendarDate = Annotated[date, BeforeValidator(parse_date)]
Amount = Annotated[float, Field(strict=True, allow_inf_nan=False)]
RATIO_FEATURES = ("annuity_to_reported_income", "credit_amount_to_reported_income")
EMPLOYMENT_INDUSTRY_FEATURE = "person_1_applicant_empl_industry_691L"
INCOME_TYPE_FEATURE = "person_1_applicant_incometype_1044T"


class RawApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_date: CalendarDate
    annuity_amount: Amount | None
    credit_amount: Amount | None
    employment_industry: str | None
    income_type: str | None
    main_occupation_income: Amount | None
    employment_start_date: CalendarDate | None
    debit_card_opening_dates: list[CalendarDate | None] = Field(default_factory=list)
    prior_application_dates: list[CalendarDate | None] = Field(default_factory=list)
    bureau_update_dates: list[CalendarDate | None] = Field(default_factory=list)


class Prediction(BaseModel):
    default_probability: float


class PredictionOptions(BaseModel):
    employment_industries: list[str]
    income_types: list[str]


def application_features(application):
    """Map one application's source fields into the shared DuckDB transforms."""
    dates_by_source = {
        "debitcard": application.debit_card_opening_dates,
        "applprev": application.prior_application_dates,
        "credit_bureau_a": application.bureau_update_dates,
    }
    with duckdb.connect(config={"threads": 1}) as connection:
        cases = connection.from_arrow(pa.table({
            "case_id": [0],
            "date_decision": pa.array([application.decision_date], type=pa.date32()),
            "static_0_annuity_780A": pa.array([application.annuity_amount], type=pa.float64()),
            "static_0_credamount_770A": pa.array([application.credit_amount], type=pa.float64()),
            EMPLOYMENT_INDUSTRY_FEATURE: pa.array([application.employment_industry], type=pa.string()),
            INCOME_TYPE_FEATURE: pa.array([application.income_type], type=pa.string()),
            "person_1_applicant_mainoccupationinc_384A": pa.array([application.main_occupation_income], type=pa.float64()),
            "person_1_applicant_empl_employedfrom_271D": pa.array([application.employment_start_date], type=pa.date32()),
        }))
        histories = []
        for source, date_column, prefix in HISTORY_DATE_SOURCES:
            if source not in dates_by_source:
                continue
            dates = dates_by_source[source]
            history = connection.from_arrow(pa.table({
                "case_id": pa.array([0] * len(dates), type=pa.int64()),
                date_column: pa.array(dates, type=pa.date32()),
            }))
            histories.append((history, date_column, prefix))
        return add_history_features(clean_table(cases, report=False), histories).df()


def registered_model_metadata(client, model_name, alias):
    """Resolve the alias once and bind the model and input contract to that version."""
    version = client.get_model_version_by_alias(model_name, alias)
    if version.status != "READY" or not version.run_id:
        raise ValueError(f"{model_name}@{alias} must point to a ready model version with a source run.")
    inputs = read_model_input_metadata(client, version.run_id)
    return {**inputs, "model_uri": f"models:/{version.name}/{version.version}"}


@asynccontextmanager
async def lifespan(app):
    model_name = os.environ.get("MODEL_NAME", "credit-risk")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    frozen = registered_model_metadata(mlflow.MlflowClient(), model_name, alias)
    check_scoring_runtime(frozen)
    model = load_frozen_booster(frozen)
    empty = RawApplication(
        decision_date=date(2000, 1, 1), annuity_amount=None, credit_amount=None,
        employment_industry=None, income_type=None, main_occupation_income=None, employment_start_date=None,
    )
    frame = application_features(empty).drop(columns=[
        "case_id", "date_decision", "person_1_applicant_empl_employedfrom_271D",
    ])
    missing = set(frozen["features"]).difference(frame.columns)
    if missing:
        raise ValueError(f"The raw application schema cannot produce model features {sorted(missing)}.")
    prepared, _ = prepare_model_inputs(frame, frozen)
    predict_probabilities(model, prepared, threads=1)
    app.state.frozen = frozen
    app.state.model = model
    app.state.prediction_lock = Lock()
    logging.getLogger("uvicorn.error").info("Loaded %s@%s from %s", model_name, alias, frozen["model_uri"])
    yield


app = FastAPI(title="Credit Risk Prediction", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_application(request: Request, error: RequestValidationError):
    details = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/predict/options")
def prediction_options(request: Request) -> PredictionOptions:
    levels = request.app.state.frozen["levels"]
    return PredictionOptions(
        employment_industries=levels.get(EMPLOYMENT_INDUSTRY_FEATURE, []),
        income_types=levels.get(INCOME_TYPE_FEATURE, []),
    )


@app.post("/predict")
def predict(application: RawApplication, request: Request) -> Prediction:
    state = request.app.state
    frame = application_features(application)
    if any(isinf(frame.at[0, name]) for name in RATIO_FEATURES):
        raise HTTPException(status_code=422, detail="Amounts and income must produce finite affordability ratios.")
    prepared, _ = prepare_model_inputs(frame, state.frozen)
    with state.prediction_lock:
        [probability] = predict_probabilities(state.model, prepared, threads=1)
    return Prediction(default_probability=float(probability))
