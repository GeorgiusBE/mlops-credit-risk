"""Check the UI API that is not exercised by normal MLflow tracking calls."""

from mlflow.server import app


def test_graphql_endpoint_returns_json():
    """The UI must receive GraphQL JSON rather than an HTML error page."""
    response = app.test_client().post("/graphql", json={"query": "{ __typename }"})
    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.is_json
    payload = response.get_json()
    assert not payload.get("errors")
    assert payload["data"]["__typename"] == "Query"
