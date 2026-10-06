from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heart_disease.data_pipeline import build_model_pipeline, load_dataset, prepare_training_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart+disease" / "processed.cleveland.data"


def test_build_model_pipeline_returns_classifier_pipeline():
    model = LogisticRegression(max_iter=1000)
    pipeline = build_model_pipeline(model)

    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_model_pipeline_can_fit_and_predict():
    frame = load_dataset(DATA_PATH)
    X, y = prepare_training_data(frame)

    pipeline = build_model_pipeline(RandomForestClassifier(n_estimators=50, random_state=42))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape[0] == X.shape[0]
    assert set(np.unique(predictions)).issubset({0, 1})
