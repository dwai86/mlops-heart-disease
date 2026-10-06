# Heart Disease Prediction MLOps Report

## 1. Objective
This project addresses the challenge of predicting heart disease from patient clinical measurements using a production-ready MLOps workflow. The solution includes data cleaning, feature preparation, model selection, experiment tracking, automated testing, API serving, monitoring, and deployment templates.

## 2. Dataset and problem definition
The project uses the UCI Cleveland heart disease dataset stored under `data/raw/heart+disease/processed.cleveland.data`. The target is converted into a binary classification problem:
- 0 = no heart disease
- 1 = heart disease present

The raw dataset includes missing values represented as `?`, which are handled during model training with a pipeline-based median imputer to avoid leakage from the full dataset into the validation process.

## 3. Exploratory data analysis
EDA is performed to understand the target distribution, feature spread, and relationships among variables. The notebook and screenshot assets include:
- class-balance bar chart
- histograms for major clinical variables
- correlation heatmap
- missing-value scan

The project has generated screenshot artifacts under `screenshots/` to support the final reporting and review process.

## 4. Feature engineering and preprocessing
The training pipeline keeps the feature matrix intact while applying median imputation and z-score normalization within the model pipeline itself. This ensures the model learns from only the training data and validation metrics remain trustworthy. The selected feature set is the widely-used 13-variable subset from the Cleveland dataset.

## 5. Model development and tuning
Two candidate classifiers are used:
1. Logistic Regression
2. Random Forest

The project performs a cross-validation-based grid search for each model to tune parameters such as regularization strength, solver choice, number of trees, tree depth, and minimum leaf size. Model performance is assessed using:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC

The best-performing model is selected according to ROC-AUC on the held-out validation set.

## 6. Experiment tracking with MLflow
All model experiments are logged to MLflow. This includes:
- model parameters
- best hyperparameters
- evaluation metrics
- confusion-matrix plots
- saved model artifacts

The logs are stored in a local MLflow tracking directory and are ideal for comparing runs across iterations.

## 7. Model packaging and reproducibility
The project creates a reusable serialization pipeline using joblib. The saved model includes the preprocessing steps, guaranteeing reproducible inference on raw feature dictionaries. The repository also provides installation instructions and a Python package configuration to simplify setup in a fresh environment.

## 8. API and serving architecture
The service is implemented with FastAPI and exposes the following endpoints:
- `/health` — service readiness status
- `/predict` — prediction and confidence output
- `/metrics` — Prometheus-formatted metrics

The API accepts patient records in JSON format and returns a prediction label, probability, and risk category.

## 9. CI/CD and automated validation
GitHub Actions is configured to run the required checks automatically. The workflow now performs:
- dependency installation
- linting with Ruff
- unit tests
- model training

This enforces a consistent validation loop for each code change and supports earlier detection of regressions.

## 10. Deployment and monitoring
The repository includes:
- a Dockerfile for packaging the service
- a Kubernetes deployment manifest with service exposure
- a Prometheus config for scraping metrics

These assets provide the structure required for local or cloud deployment. Monitoring is done through Prometheus-compatible metrics exposed by the FastAPI service.

## 11. Architecture diagram
```text
User / Client
      |
      v
FastAPI app (/predict, /health, /metrics)
      |
      v
Preprocessing pipeline + trained model artifact
      |
      +--> MLflow experiment tracking
      +--> CI/CD validation (GitHub Actions)
      +--> Container / Kubernetes deployment
      +--> Prometheus monitoring
```

## 12. Conclusion
This project demonstrates a complete MLOps workflow for a healthcare classification problem. It covers data preparation, EDA, model tuning, experiment tracking, test automation, API serving, deployment templates, and operational monitoring. The repository is structured to be robust, reproducible, and suitable for academic submission and further extension in real-world production settings.
