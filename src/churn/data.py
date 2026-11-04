"""Carregamento e limpeza do dataset Telco."""

from pathlib import Path

import pandas as pd

from churn.config import FEATURES, ID_COLUMN, RAW_DATA_PATH, TARGET


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Rode `uv run churn-download` primeiro."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza determinística, aplicada antes do pipeline do Scikit-Learn.

    - TotalCharges vem como texto, com espaços em branco para clientes com tenure 0
      (ainda não faturados): converte para número e preenche esses casos com 0.
    - Remove o identificador do cliente.
    - Converte o target Yes/No para 1/0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    not_billed_yet = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[not_billed_yet, "TotalCharges"] = 0.0

    if ID_COLUMN in df.columns:
        df = df.drop(columns=ID_COLUMN)
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET].astype(int)


def load_dataset(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(clean(load_raw(path)))
