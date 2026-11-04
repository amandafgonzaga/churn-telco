"""Pré-processamento como ColumnTransformer, para viver dentro do Pipeline do modelo.

Manter o pré-processamento dentro do Pipeline garante que:
- na validação cruzada, imputer/scaler/encoder são ajustados só no fold de treino (sem leakage);
- o artefato salvo (.joblib) recebe dados crus — a API não reimplementa nenhuma transformação.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # essencial para Regressão Logística e MLP
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
