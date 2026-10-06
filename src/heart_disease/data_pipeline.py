from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"
DATASET_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    frame = pd.read_csv(dataset_path, header=None, na_values="?")

    if frame.shape[1] != len(DATASET_COLUMNS):
        raise ValueError(
            f"Unexpected dataset shape: expected {len(DATASET_COLUMNS)} columns, "
            f"got {frame.shape[1]}."
        )

    frame.columns = DATASET_COLUMNS
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(float)
    frame[TARGET_COLUMN] = (frame[TARGET_COLUMN] > 0).astype(int)
    return frame


def prepare_training_data(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    frame = frame.copy()
    frame[TARGET_COLUMN] = (frame[TARGET_COLUMN] > 0).astype(int)
    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET_COLUMN].astype(int)
    return X, y


def build_model_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def run_training(data_path: str | Path, artifact_dir: str | Path, tracking_uri: str = "file:./mlruns") -> Dict[str, Dict[str, float]]:
    dataset_path = Path(data_path)
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    X, y = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-prediction")

    model_search_spaces = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear", "lbfgs"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "param_grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
    }

    best_name = None
    best_model = None
    best_metrics = None
    best_cv_metrics = None
    best_cv_score = None
    model_results: Dict[str, Dict[str, float]] = {}

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    for model_name, config in model_search_spaces.items():
        model_pipeline = build_model_pipeline(config["estimator"])
        search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=config["param_grid"],
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=-1,
            refit="roc_auc",
            return_train_score=False,
        )

        with mlflow.start_run(run_name=model_name):
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_index = search.best_index_
            metrics = evaluate_model(best_pipeline, X_test, y_test)

            cv_metrics = {
                "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][best_index]),
                "cv_accuracy_std": float(search.cv_results_["std_test_accuracy"][best_index]),
                "cv_precision_mean": float(search.cv_results_["mean_test_precision"][best_index]),
                "cv_precision_std": float(search.cv_results_["std_test_precision"][best_index]),
                "cv_recall_mean": float(search.cv_results_["mean_test_recall"][best_index]),
                "cv_recall_std": float(search.cv_results_["std_test_recall"][best_index]),
                "cv_f1_mean": float(search.cv_results_["mean_test_f1"][best_index]),
                "cv_f1_std": float(search.cv_results_["std_test_f1"][best_index]),
                "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][best_index]),
                "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best_index]),
            }

            model_results[model_name] = {
                **cv_metrics,
                **{f"test_{key}": float(value) for key, value in metrics.items()},
            }

            mlflow.log_params({"model_name": model_name, **search.best_params_})
            mlflow.log_metrics(
                {
                    **cv_metrics,
                    **{f"test_{key}": float(value) for key, value in metrics.items()},
                }
            )

            plot_path = artifact_path / f"{model_name}_metrics.png"
            plot_metric_summary(
                plot_path,
                y_test,
                best_pipeline.predict(X_test),
                best_pipeline.predict_proba(X_test)[:, 1],
            )
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            joblib_path = artifact_path / f"{model_name}.joblib"
            joblib.dump(best_pipeline, joblib_path)
            mlflow.log_artifact(str(joblib_path), artifact_path="models")

            cv_roc_auc = cv_metrics["cv_roc_auc_mean"]
            if best_cv_score is None or cv_roc_auc > best_cv_score:
                best_name = model_name
                best_model = best_pipeline
                best_metrics = metrics
                best_cv_metrics = cv_metrics
                best_cv_score = cv_roc_auc

    if best_model is None or best_name is None:
        raise RuntimeError("No trained model could be produced.")

    final_model_path = artifact_path / "best_model.joblib"
    joblib.dump(best_model, final_model_path)
    with mlflow.start_run(run_name="best-model"):
        mlflow.log_params({
            "selected_model": best_name,
            "selection_metric": "cv_roc_auc_mean",
            "cv_folds": 5,
        })

        mlflow.log_metrics(
            {
                **best_cv_metrics,
                **{
                    f"test_{key}": float(value)
                    for key, value in best_metrics.items()
                },
            }
        )

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    return model_results


def plot_metric_summary(path: str | Path, y_true, y_pred, y_prob) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", color="black")

    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.7, label="No disease")
    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.7, label="Disease")
    axes[1].set_title("Predicted probability distribution")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def predict_record(model, patient_record: dict) -> dict:
    row = pd.DataFrame([patient_record], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(model.predict(row)[0])
    return {"prediction": prediction, "probability": probability, "risk_label": "High risk" if prediction == 1 else "Low risk"}
