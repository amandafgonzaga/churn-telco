import pandas as pd

from churn.config import FEATURES, ID_COLUMN, TARGET
from churn.data import clean, split_features_target


def test_clean_converts_blank_total_charges_to_zero(raw_df):
    cleaned = clean(raw_df)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert (cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"] == 0).all()


def test_clean_maps_target_and_drops_id(raw_df):
    cleaned = clean(raw_df)

    assert ID_COLUMN not in cleaned.columns
    assert set(cleaned[TARGET].unique()) <= {0, 1}


def test_clean_does_not_mutate_input(raw_df):
    original = raw_df.copy()
    clean(raw_df)

    pd.testing.assert_frame_equal(raw_df, original)


def test_split_features_target_returns_expected_columns(raw_df):
    X, y = split_features_target(clean(raw_df))

    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(raw_df)
