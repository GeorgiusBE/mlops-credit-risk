"""Enter a raw credit application and score it through the prediction API."""

from datetime import date
import os

import requests
import streamlit as st


API_URL = os.environ.get("PREDICTION_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_json(method, path, **kwargs):
    response = requests.request(method, f"{API_URL}{path}", timeout=(2, 15), **kwargs)
    if response.status_code != 422:
        response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("The prediction API returned an unexpected response.")
    if response.status_code == 422:
        detail = body.get("detail")
        raise ValueError(detail if isinstance(detail, str) else "Check the application fields and try again.")
    return body


def history_dates(text, field):
    dates = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is None or parsed.isoformat() != value:
            raise ValueError(f"{field}, line {line_number}. Enter a date in YYYY-MM-DD format.")
        dates.append(value)
    return dates


st.set_page_config(page_title="Credit risk prediction", page_icon=":material/analytics:", layout="wide")
st.title("Credit risk prediction")
st.caption("Estimate the probability of default for a single application.")

try:
    options = api_json("GET", "/predict/options")
    industries = options["employment_industries"]
    income_types = options["income_types"]
    if not isinstance(industries, list) or not isinstance(income_types, list):
        raise ValueError("The prediction API returned invalid category options.")
except (requests.RequestException, ValueError, KeyError):
    st.error("Start FastAPI with `make serve`, then reload this page.", title="Prediction API unavailable")
    st.stop()

form_column, result_column = st.columns([3, 2], gap="large")
with form_column, st.form("application", enter_to_submit=False):
    st.subheader("Application details")
    st.caption("Use the source amounts and leave unavailable fields blank.")

    decision_field, credit_field, annuity_field = st.columns(3)
    with decision_field:
        decision_date = st.date_input(
            "Decision date", value="today", min_value=date.min, max_value=date.max, format="YYYY-MM-DD",
            key="decision_date", help="Use the date the estimate is for. Only earlier history events count.",
        )
    with credit_field:
        credit_amount = st.number_input("Credit amount", value=None, step=1.0, key="credit_amount")
    with annuity_field:
        annuity_amount = st.number_input("Annuity amount", value=None, step=1.0, key="annuity_amount")

    st.subheader("Employment")
    left, right = st.columns(2)
    with left:
        employment_industry = st.selectbox(
            "Employment industry", industries, index=None, key="employment_industry",
        )
        employment_start_date = st.date_input(
            "Employment start date", value=None, min_value=date.min, max_value=date.max, format="YYYY-MM-DD",
            key="employment_start_date",
        )
    with right:
        income_type = st.selectbox("Income type", income_types, index=None, key="income_type")
        main_occupation_income = st.number_input(
            "Main occupation income", value=None, step=1.0, key="main_occupation_income",
        )

    with st.expander("History dates", icon=":material/history:"):
        st.caption("Enter one date per line in YYYY-MM-DD format. Leave empty if no records are available.")
        debit_card_dates = st.text_area("Debit card opening dates", height=88, key="debit_card_opening_dates")
        prior_application_dates = st.text_area("Prior application dates", height=88, key="prior_application_dates")
        bureau_dates = st.text_area("Active bureau update dates", height=88, key="bureau_update_dates")

    submitted = st.form_submit_button("Estimate default probability", type="primary", icon=":material/analytics:")

with result_column, st.container(border=True):
    if not submitted:
        st.metric("Probability of default", "—")
        st.caption("Submit the application to see the estimate here.")
    else:
        try:
            if decision_date is None:
                raise ValueError("Enter a decision date.")
            payload = {
                "decision_date": decision_date.isoformat(),
                "annuity_amount": annuity_amount,
                "credit_amount": credit_amount,
                "employment_industry": employment_industry,
                "income_type": income_type,
                "main_occupation_income": main_occupation_income,
                "employment_start_date": employment_start_date.isoformat() if employment_start_date else None,
                "debit_card_opening_dates": history_dates(debit_card_dates, "Debit card opening dates"),
                "prior_application_dates": history_dates(prior_application_dates, "Prior application dates"),
                "bureau_update_dates": history_dates(bureau_dates, "Active bureau update dates"),
            }
            with st.spinner("Scoring application…"):
                probability = api_json("POST", "/predict", json=payload).get("default_probability")
            if type(probability) not in (int, float) or not 0 <= probability <= 1:
                raise ValueError("The prediction API returned an invalid probability.")
        except requests.RequestException:
            st.error("Check that FastAPI is running and submit again.", title="Prediction unavailable")
        except ValueError as error:
            st.error(str(error))
        else:
            st.metric("Probability of default", f"{probability:.2%}")
