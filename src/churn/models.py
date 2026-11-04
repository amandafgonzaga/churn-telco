"""Fábrica dos modelos comparados: linear (baseline), árvore (ensemble) e rede neural."""

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn.config import SEED
from churn.features import build_preprocessor


def get_models(seed: int = SEED) -> dict[str, ClassifierMixin]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        ),
        # MLPClassifier não aceita class_weight: o desbalanceamento é compensado
        # pelo ajuste de threshold no treino (ver train.py).
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=1e-3,
            early_stopping=True,
            max_iter=500,
            random_state=seed,
        ),
    }


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
