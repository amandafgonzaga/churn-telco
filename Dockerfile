# Imagem da API de inferência (deploy opcional: Cloud Run, App Runner, Azure Container Apps...)
# Requer models/champion.joblib já treinado.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

COPY models ./models

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-dev uvicorn churn.api.main:app --host 0.0.0.0 --port ${PORT}"]
