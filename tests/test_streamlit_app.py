"""Check the prediction form's request and response through Streamlit widgets."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import requests
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_form_posts_raw_inputs_and_displays_probability(monkeypatch):
    def api(method, url, **kwargs):
        response = Mock(spec=requests.Response, status_code=200)
        if method == "GET":
            assert url.endswith("/predict/options")
            response.json.return_value = {"employment_industries": ["IT"], "income_types": ["EMPLOYED"]}
        else:
            assert url.endswith("/predict")
            assert kwargs["json"] == {
                "decision_date": "2026-10-01",
                "credit_amount": 38000.0,
                "annuity_amount": None,
                "employment_industry": "IT",
                "income_type": "EMPLOYED",
                "main_occupation_income": None,
                "employment_start_date": "2021-04-12",
                "debit_card_opening_dates": ["2020-06-12", "2022-01-01"],
                "prior_application_dates": [],
                "bureau_update_dates": [],
            }
            response.json.return_value = {"default_probability": 0.06655709086973012}
        return response

    request = Mock(side_effect=api)
    monkeypatch.setattr(requests, "request", request)
    app = AppTest.from_file(str(APP)).run()
    assert app.date_input(key="decision_date").value == date.today()
    assert app.metric[0].value == "—"

    app.text_area(key="debit_card_opening_dates").set_value("not a date")
    app.button[0].click().run()
    assert not app.exception
    assert "line 1" in app.error[0].value
    assert all(call.args[0] == "GET" for call in request.call_args_list)

    app.date_input(key="decision_date").set_value(date(2026, 10, 1))
    app.number_input(key="credit_amount").set_value(38000.0)
    app.selectbox(key="employment_industry").select("IT")
    app.selectbox(key="income_type").select("EMPLOYED")
    app.date_input(key="employment_start_date").set_value(date(2021, 4, 12))
    app.text_area(key="debit_card_opening_dates").set_value("2020-06-12\n\n2022-01-01")
    app.button[0].click().run()

    assert not app.exception
    assert not app.error
    assert app.metric[0].label == "Probability of default"
    assert app.metric[0].value == "6.66%"


def test_form_explains_when_prediction_api_is_unavailable(monkeypatch):
    monkeypatch.setattr(requests, "request", Mock(side_effect=requests.ConnectionError))
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert "make serve" in app.error[0].value
