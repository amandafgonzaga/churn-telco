"""API de inferência de churn.

Uso:
    uv run uvicorn churn.api.main:app --reload
    Documentação interativa: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Request

from churn import __version__
from churn.api.schemas import (
    PREDICT_BATCH_EXAMPLES,
    PREDICT_EXAMPLES,
    CustomerIn,
    HealthOut,
    PredictionOut,
)
from churn.predict import ChurnModel, load_model

logger = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Os testes injetam um modelo em app.state.model antes de subir a aplicação.
    if getattr(app.state, "model", None) is None:
        try:
            app.state.model = load_model()
            logger.info("Modelo carregado: %s", app.state.model.version)
        except FileNotFoundError as exc:
            logger.warning("API iniciada sem modelo: %s", exc)
            app.state.model = None
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Propensão de cancelamento de clientes de telecomunicações.",
    version=__version__,
    lifespan=lifespan,
)


def get_model(request: Request) -> ChurnModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return model


@app.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    model = getattr(request.app.state, "model", None)
    return HealthOut(
        status="ok",
        model_loaded=model is not None,
        model_version=model.version if model else None,
    )


LoadedModel = Annotated[ChurnModel, Depends(get_model)]


@app.post("/predict", response_model=PredictionOut)
def predict(
    customer: Annotated[CustomerIn, Body(openapi_examples=PREDICT_EXAMPLES)],
    model: LoadedModel,
) -> PredictionOut:
    return PredictionOut(**model.predict([customer.model_dump()])[0])


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(
    customers: Annotated[list[CustomerIn], Body(openapi_examples=PREDICT_BATCH_EXAMPLES)],
    model: LoadedModel,
) -> list[PredictionOut]:
    if not customers:
        return []
    return [PredictionOut(**r) for r in model.predict([c.model_dump() for c in customers])]
