# syntax=docker/dockerfile:1
FROM python:3.14.4-slim-trixie AS dependencies
WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install poetry==2.4.1 && python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only main --no-root --no-interaction

FROM python:3.14.4-slim-trixie AS runtime
WORKDIR /app

# LightGBM's Linux wheel uses the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY --from=dependencies /opt/venv /opt/venv
COPY prediction_api.py streamlit_app.py ./
COPY .streamlit/config.toml .streamlit/config.toml
COPY data_preprocessing/ data_preprocessing/
COPY feature_selection/ feature_selection/
COPY model_evaluation/ model_evaluation/

USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
