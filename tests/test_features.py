import numpy as np

from churn.features import build_preprocessor
from churn.models import build_pipeline, get_models


def test_preprocessor_output_has_no_nan_and_keeps_rows(train_xy):
    X, _ = train_xy
    X = X.copy()
    X.loc[X.index[0], "TotalCharges"] = np.nan  # o imputer deve cobrir

    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()


def test_preprocessor_ignores_unknown_category(train_xy):
    X, _ = train_xy
    preprocessor = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["PaymentMethod"] = "Pix"

    assert preprocessor.transform(unseen).shape[0] == 1


def test_training_is_reproducible_with_fixed_seed(train_xy):
    X, y = train_xy
    first = build_pipeline(get_models()["random_forest"]).fit(X, y).predict_proba(X)
    second = build_pipeline(get_models()["random_forest"]).fit(X, y).predict_proba(X)

    np.testing.assert_array_equal(first, second)
