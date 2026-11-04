import numpy as np

from churn.config import COST_RETENTION_OFFER, LOST_REVENUE_MONTHS
from churn.evaluate import best_threshold, business_cost, classification_metrics


def test_business_cost_counts_offers_and_missed_churn():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.2, 0.8, 0.1])  # 1 churn detectado, 1 perdido, 1 falso positivo
    monthly = np.array([100.0, 50.0, 70.0, 30.0])

    cost = business_cost(y_true, y_proba, monthly, threshold=0.5)

    assert cost == 2 * COST_RETENTION_OFFER + 50.0 * LOST_REVENUE_MONTHS


def test_best_threshold_beats_doing_nothing():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.8, 0.7, 0.3, 0.2, 0.1, 0.4])
    monthly = np.full(6, 80.0)

    threshold, cost = best_threshold(y_true, y_proba, monthly)

    assert 0.4 <= threshold <= 0.7
    assert cost < business_cost(y_true, y_proba, monthly, threshold=1.0)


def test_classification_metrics_perfect_model():
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
