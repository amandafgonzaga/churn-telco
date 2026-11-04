"""Métricas técnicas e métrica de negócio (custo do churn)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import COST_RETENTION_OFFER, LOST_REVENUE_MONTHS

# Métricas usadas na validação cruzada (nomes do sklearn.metrics.get_scorer)
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}
PRIMARY_METRIC = "roc_auc"


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


def business_cost(y_true, y_proba, monthly_charges, threshold: float) -> float:
    """Custo total da política "oferecer retenção a quem o modelo sinaliza".

    Premissa simplificada: a oferta (custo fixo) vai para todo cliente sinalizado e
    churns não detectados perdem LOST_REVENUE_MONTHS meses de mensalidade.
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(y_proba) >= threshold
    missed_churn = (y_true == 1) & ~flagged
    offers_cost = flagged.sum() * COST_RETENTION_OFFER
    lost_revenue = (np.asarray(monthly_charges)[missed_churn] * LOST_REVENUE_MONTHS).sum()
    return float(offers_cost + lost_revenue)


def best_threshold(
    y_true, y_proba, monthly_charges, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Retorna (threshold, custo) que minimiza o custo de negócio."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    costs = [business_cost(y_true, y_proba, monthly_charges, t) for t in grid]
    idx = int(np.argmin(costs))
    return round(float(grid[idx]), 4), float(costs[idx])
