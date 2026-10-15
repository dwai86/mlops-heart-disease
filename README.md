# Heart Disease MLOps Project

This repository implements an end-to-end machine learning operations pipeline for predicting heart disease risk from the UCI Cleveland heart disease dataset. The project includes data preparation, feature engineering, model training with MLflow, API serving with FastAPI, Docker packaging, CI automation, local Kubernetes deployment YAML, and monitoring configuration.

## Project structure

- `data/raw/heart+disease/` — downloaded raw UCI dataset files
- `scripts/download_data.py` — fetches or validates the dataset
- `scripts/train_model.py` — trains and logs the ML models
- `src/heart_disease/` — reusable training and API code
- `tests/` — unit tests for data processing and modeling
- `.github/workflows/ci.yml` — GitHub Actions pipeline for lint/test/train
- `Dockerfile` — container for the model-serving API
- `kubernetes/deployment.yaml` — two-replica Deployment with a LoadBalancer Service
- `kubernetes/local-nodeport.yaml` — single-replica local Deployment with a NodePort Service and readiness probe
- `monitoring/prometheus.yml` — Prometheus scraping config
- `docs/Heart_Disease_MLOps_Interactive_Report.html` — interactive report source; export a PDF for submission
- `screenshots/` — EDA charts, architecture diagrams, and Kubernetes deployment screenshots
  - `screenshots/bits lab evidence/` — BITS lab test, training, Docker build, and API validation evidence

## Dataset

The project uses the Cleveland subset of the heart disease UCI dataset. The required dataset is stored in `data/raw/heart+disease/processed.cleveland.data`.

## Quick start

1. Activate the virtual environment:
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies and make the package importable:
   ```bash
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```
   If you prefer not to install editable mode, set `PYTHONPATH=src` before running Python commands.
3. Train the models:
   ```bash
   python scripts/train_model.py --data data/raw/heart+disease/processed.cleveland.data --artifact-dir artifacts
   ```
4. Start the API locally:
   ```bash
   set PYTHONPATH=src
   uvicorn heart_disease.api:app --host 0.0.0.0 --port 8000
   ```
5. Test the prediction endpoint:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
   ```

## Model training and evaluation

The training pipeline loads the Cleveland dataset, converts the target into a binary classification problem (0 = no disease, 1 = disease), imputes missing values, standardizes numeric inputs, and imputes and one-hot encodes categorical inputs within the model pipeline. It fits both logistic regression and random forest classifiers. Model performance is evaluated with accuracy, precision, recall, F1-score, and ROC-AUC. Metrics are logged to MLflow at `mlruns/`.

## Architecture

```text
User / Client
    |
    v
FastAPI service (/predict)
    |
    v
Model artifact (best_model.joblib)
    |
    v
Preprocessing + trained classifier
    |
    +--> MLflow tracking (parameters + metrics + plots)
    +--> Prometheus metrics (/metrics)
```

## CI/CD and deployment

- GitHub Actions automatically installs dependencies, runs the unit tests, and executes the model training job.
- Docker builds the API container locally or in CI.
- `kubernetes/deployment.yaml` provides a two-replica LoadBalancer configuration; `kubernetes/local-nodeport.yaml` is the local NodePort configuration used for the single-replica Kubernetes health-check evidence in `screenshots/Kubernetes_Deployment1.png` and `screenshots/Kubernetes_Deployment2.png`.
- Prometheus scrapes `/metrics` for request counts and latency monitoring.

## Testing

Run the test suite with:

```bash
pytest -q
```

## Documentation

Open the [interactive final report](docs/Heart_Disease_MLOps_Interactive_Report.html)
for EDA, model development, experiment tracking, deployment, monitoring,
and the current architecture diagram. The standalone updated diagram is
available as [SVG](<screenshots/Heart Disease Prediction MLOps Architecture.svg>).
Export and inspect a PDF copy before submission; the PDF is not generated in
the repository yet.

### Video demonstration

[Watch the project walkthrough](https://drive.google.com/file/d/10s3rjoWsSsPucqV7k0s7a4kNOKY1ZHZJ/view?usp=sharing).
