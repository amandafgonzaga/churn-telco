"""Carregamento do modelo campeão e inferência."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import CHAMPION_PATH, DEFAULT_THRESHOLD, FEATURES


@dataclass
class ChurnModel:
    pipeline: Pipeline
    threshold: float = DEFAULT_THRESHOLD
    version: str = "unknown"

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        X = pd.DataFrame.from_records(records, columns=FEATURES)
        probas = self.pipeline.predict_proba(X)[:, 1]
        return [
            {
                "churn_probability": round(float(p), 4),
                "churn_prediction": bool(p >= self.threshold),
                "threshold": self.threshold,
                "model_version": self.version,
            }
            for p in probas
        ]


def metadata_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def load_model(model_path: Path | None = None) -> ChurnModel:
    """Carrega o pipeline salvo e os metadados (threshold, versão) ao lado dele.

    O caminho pode ser sobrescrito pela variável de ambiente CHURN_MODEL_PATH.
    """
    model_path = Path(model_path or os.getenv("CHURN_MODEL_PATH", CHAMPION_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode `uv run churn-train` primeiro."
        )
    pipeline = joblib.load(model_path)

    meta_path = metadata_path_for(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return ChurnModel(
        pipeline=pipeline,
        threshold=metadata.get("threshold", DEFAULT_THRESHOLD),
        version=metadata.get("version", "unknown"),
    )
