from pathlib import Path

import pandas as pd

from heart_disease.data_pipeline import FEATURE_COLUMNS, load_dataset, prepare_training_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart+disease" / "processed.cleveland.data"


def test_load_dataset_has_expected_columns():
    frame = load_dataset(DATA_PATH)

    assert frame.shape[0] > 0
    assert frame.shape[1] == 14
    assert set(frame.columns) == set(FEATURE_COLUMNS + ["target"])
    assert set(frame["target"].unique()).issubset({0, 1})


def test_prepare_training_data_handles_missing_values():
    frame = load_dataset(DATA_PATH)
    X, y = prepare_training_data(frame)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert not pd.isna(X).any().any()
    assert set(y.unique()).issubset({0, 1})
