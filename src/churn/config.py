"""Configurações centrais: seed, caminhos, features e parâmetros de negócio."""

from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_PATH = MODELS_DIR / "champion.joblib"
EXPERIMENTS_PATH = PROJECT_ROOT / "docs" / "experimentos.csv"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.5

# Custo de negócio (premissas a validar com a área de negócio — ver docs/ml_canvas.md):
# - cada cliente sinalizado recebe uma oferta de retenção que custa COST_RETENTION_OFFER;
# - cada churn não detectado (falso negativo) perde MonthlyCharges * LOST_REVENUE_MONTHS.
COST_RETENTION_OFFER = 50.0
LOST_REVENUE_MONTHS = 12
