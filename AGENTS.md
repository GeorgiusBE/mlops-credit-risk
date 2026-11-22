# Repository guidance

This project trains and evaluates a LightGBM credit-risk model, serves default
probabilities through FastAPI, and collects application inputs through Streamlit.
MLflow tracks experiments and supplies the registered model and its feature
metadata at API startup. DVC versions the training inputs.

## Local workflow

- Run commands from the repository root. Follow the [README](README.md) for
  Poetry setup and supported Make targets. Use the locked dependencies.
- Use `make preprocess` for the supported preprocessing pipeline and
  `make dvc-status` to inspect the tracked data. See the
  [data versioning guide](docs/data-versioning.md) for restoring snapshots.
- Start `make mlflow`, `make serve`, and `make ui` in separate terminals to run
  the local serving stack. The API defaults to `models:/credit-risk@champion`.
- Run `make test` for code changes. For serving changes, also submit a synthetic
  application through the running stack and verify the returned probability.

## Modeling and inference

- Preserve chronological validation and reserve the held-out weeks for final
  evaluation. Base feature selection, tuning, and threshold decisions on
  development data. Follow the existing review steps in the
  [modeling workflow](docs/credit-risk-modelling-workflow.md).
- Keep the model, ordered features, dtypes, and categorical metadata tied to the
  same MLflow source run. Preserve feature calculations across training and
  inference, including the use of events strictly before the decision date.
- Treat changes to the prediction request, response, or feature contract as
  coordinated changes across the API, Streamlit form, and tests.

## Deployment roadmap

1. Run the [local Docker Compose stack](README.md#run-the-serving-stack-with-docker-compose)
   with `make compose-up` and tear it down with `make compose-down`. Verify from
   a clean checkout with a restored MLflow registry that the API loads a known
   model version and scores a synthetic application. Keep training datasets,
   the DVC cache, and credentials outside the serving image. Configure service
   endpoints at runtime.
2. Use GitHub Actions to run tests and build the serving image on pull requests.
   Keep application releases and model promotions independently reproducible.
   Record the image digest and resolved MLflow model version for each deployment.
3. For a future ECS staging deployment, document how the service obtains the
   model and feature metadata, how readiness is checked after model loading, and
   how to roll back the image or model version independently.
