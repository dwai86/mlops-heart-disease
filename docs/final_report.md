# Final Project Report

## Objective
The goal of this project is to build a reproducible heart disease prediction pipeline using the UCI Cleveland dataset, log experiments with MLflow, package the model into an API, and show a production-style deployment flow.

## Data acquisition and EDA
The dataset is located under `data/raw/heart+disease/processed.cleveland.data`. The preprocessing step removes missing values using median imputation, converts the target to a binary label, and retains the 13 clinical features used in prior medical classification experiments.

Key exploratory checks include:
- shape validation
- missing value audit
- class-balance analysis
- train/test split with stratification

## Modeling
Two classifiers are trained:
1. Logistic Regression
2. Random Forest

Both models are wrapped in preprocessing pipelines containing median imputation and feature scaling. The best model is chosen by comparing ROC-AUC. The project logs all metrics and artifacts to MLflow under the `heart-disease-prediction` experiment.

## MLflow tracking
The training script logs:
- model parameters
- evaluation metrics
- confusion-matrix plots
- trained model artifact

The default tracking URI is a local `mlruns/` directory.

## Serving API
The FastAPI app exposes a `/predict` endpoint that accepts the standard 13 clinical features and returns:
- prediction label
- probability score
- risk category

Prometheus metrics are also exposed through `/metrics`.

## Deployment workflow
The repository includes:
- a Dockerfile for runtime packaging
- a Kubernetes deployment manifest in `kubernetes/deployment.yaml`
- a Prometheus configuration file in `monitoring/prometheus.yml`

## CI/CD
GitHub Actions runs the unit tests, installs dependencies, and executes the training step. This makes the project suitable for automated validation on every push or pull request.

## Architecture diagram
```text
Client --> FastAPI service --> Trained model --> Prediction
    |                           |
    |                           +--> MLflow logs
    +--> Prometheus metrics ----> Monitoring dashboard
```

## Conclusion
This project demonstrates a complete MLOps workflow for a healthcare classification task, covering data quality checks, model selection, experiment tracking, packaging, deployment templates, and monitoring.
