"""Treina e compara os modelos com validação cruzada e salva o campeão.

Uso:
    uv run churn-train
    uv run churn-train --data caminho/para/outro.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from churn.config import (
    CHAMPION_PATH,
    CV_FOLDS,
    DEFAULT_THRESHOLD,
    EXPERIMENTS_PATH,
    RAW_DATA_PATH,
    SEED,
    TEST_SIZE,
)
from churn.data import load_dataset
from churn.evaluate import (
    CV_SCORING,
    PRIMARY_METRIC,
    best_threshold,
    business_cost,
    classification_metrics,
)
from churn.models import build_pipeline, get_models
from churn.predict import metadata_path_for


def compare_models(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    rows = []
    for name, estimator in get_models().items():
        print(f"Validação cruzada: {name} ...")
        scores = cross_validate(
            build_pipeline(estimator), X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1
        )
        row = {"model": name}
        for metric in CV_SCORING:
            row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"cv_{PRIMARY_METRIC}_mean", ascending=False)


def log_experiments(results: pd.DataFrame, run_id: str, path: Path = EXPERIMENTS_PATH) -> None:
    results = results.assign(run_id=run_id, seed=SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, mode="a", header=not path.exists(), index=False)


def train(data_path: Path = RAW_DATA_PATH, model_path: Path = CHAMPION_PATH) -> dict:
    np.random.seed(SEED)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results = compare_models(X_train, y_train, cv)
    log_experiments(results, run_id)
    print("\nComparação (validação cruzada no treino):")
    print(results.round(4).to_string(index=False))

    champion_name = results.iloc[0]["model"]
    champion = build_pipeline(get_models()[champion_name])

    # Threshold escolhido com predições out-of-fold do treino: o teste não é usado aqui.
    oof_proba = cross_val_predict(clone(champion), X_train, y_train, cv=cv, method="predict_proba")[
        :, 1
    ]
    threshold, _ = best_threshold(y_train, oof_proba, X_train["MonthlyCharges"])

    champion.fit(X_train, y_train)
    test_proba = champion.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_proba, threshold)
    test_metrics["business_cost"] = business_cost(
        y_test, test_proba, X_test["MonthlyCharges"], threshold
    )
    test_metrics["business_cost_threshold_0_5"] = business_cost(
        y_test, test_proba, X_test["MonthlyCharges"], DEFAULT_THRESHOLD
    )
    # Referência: custo de não fazer nada (nenhuma oferta, todo churn é perdido)
    test_metrics["business_cost_no_model"] = business_cost(
        y_test, np.zeros_like(test_proba), X_test["MonthlyCharges"], 1.0
    )

    metadata = {
        "version": f"{champion_name}-{run_id}",
        "model": champion_name,
        "threshold": threshold,
        "seed": SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_path, compress=3)
    metadata_path_for(model_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nCampeão: {champion_name} | threshold={threshold:.2f}")
    print(json.dumps(metadata["test_metrics"], indent=2))
    print(f"Modelo salvo em {model_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e compara modelos de churn.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=CHAMPION_PATH)
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
