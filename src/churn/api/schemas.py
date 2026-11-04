"""Contratos de entrada e saída da API.

Os campos espelham as colunas cruas do dataset Telco; os Literal fazem a API
rejeitar (HTTP 422) categorias que o modelo nunca viu.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 94.5,
    "TotalCharges": 283.5,
}

LOYAL_CUSTOMER = {
    **EXAMPLE_CUSTOMER,
    "tenure": 60,
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
}

NEW_CUSTOMER = {key: value for key, value in EXAMPLE_CUSTOMER.items() if key != "TotalCharges"}
NEW_CUSTOMER["tenure"] = 0

# Exemplos selecionáveis no Swagger (/docs), no dropdown "Examples" do request body.
PREDICT_EXAMPLES = {
    "alto_risco": {
        "summary": "Alto risco: contrato mensal, 3 meses, fibra",
        "value": EXAMPLE_CUSTOMER,
    },
    "cliente_fiel": {
        "summary": "Baixo risco: contrato de 2 anos, 60 meses",
        "value": LOYAL_CUSTOMER,
    },
    "cliente_novo": {
        "summary": "Cliente novo sem TotalCharges (o pipeline imputa)",
        "value": NEW_CUSTOMER,
    },
    "invalido": {
        "summary": "Inválido: contrato 'Weekly' → HTTP 422",
        "value": {**EXAMPLE_CUSTOMER, "Contract": "Weekly"},
    },
}

PREDICT_BATCH_EXAMPLES = {
    "alto_risco_e_fiel": {
        "summary": "Dois clientes: alto risco e fiel",
        "value": [EXAMPLE_CUSTOMER, LOYAL_CUSTOMER],
    },
}


class CustomerIn(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(
        default=None, ge=0, description="Opcional: se ausente, o pipeline imputa"
    )


class PredictionOut(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    threshold: float
    model_version: str


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None = None
