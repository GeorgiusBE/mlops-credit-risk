"""Check the UI API that is not exercised by normal MLflow tracking calls."""

import unittest

from mlflow.server import app


class MLflowUITest(unittest.TestCase):
    """Catch incompatible GraphQL dependencies before opening an experiment."""

    def test_graphql_endpoint_returns_json(self):
        """The UI must receive GraphQL JSON rather than an HTML error page."""
        response = app.test_client().post("/graphql", json={"query": "{ __typename }"})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertFalse(payload.get("errors"))
        self.assertEqual(payload["data"]["__typename"], "Query")


if __name__ == "__main__":
    unittest.main()
