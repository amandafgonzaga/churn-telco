"""Fixtures com dados sintéticos no formato cru do Telco.

Os testes não dependem do CSV real nem do models/champion.joblib, então rodam em CI.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn.api.main import app
from churn.api.schemas import EXAMPLE_CUSTOMER
from churn.config import SEED
from churn.data import clean, split_features_target
from churn.models import build_pipeline, get_models
from churn.predict import ChurnModel


def make_raw_customers(n: int = 200, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = ["Yes", "No"]
    addon = ["Yes", "No", "No internet service"]
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # igual ao dataset real: em branco para quem ainda não pagou
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    churn_prob = np.where(contract == "Month-to-month", 0.45, 0.1)
    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n),
            "tenure": tenure,
            "PhoneService": rng.choice(yes_no, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(addon, n),
            "OnlineBackup": rng.choice(addon, n),
            "DeviceProtection": rng.choice(addon, n),
            "TechSupport": rng.choice(addon, n),
            "StreamingTV": rng.choice(addon, n),
            "StreamingMovies": rng.choice(addon, n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(yes_no, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(rng.random(n) < churn_prob, "Yes", "No"),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_customers()


@pytest.fixture
def train_xy(raw_df) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(clean(raw_df))


@pytest.fixture
def churn_model(train_xy) -> ChurnModel:
    X, y = train_xy
    pipeline = build_pipeline(get_models()["logistic_regression"]).fit(X, y)
    return ChurnModel(pipeline=pipeline, threshold=0.5, version="test")


@pytest.fixture
def client(churn_model):
    app.state.model = churn_model
    with TestClient(app) as c:
        yield c
    app.state.model = None


@pytest.fixture
def customer_payload() -> dict:
    return dict(EXAMPLE_CUSTOMER)
