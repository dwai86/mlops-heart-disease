from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

from heart_disease.data_pipeline import FEATURE_COLUMNS, predict_record, run_training

APP_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = APP_DIR / "artifacts" / "best_model.joblib"
DATA_PATH = APP_DIR / "data" / "raw" / "heart+disease" / "processed.cleveland.data"
LOG_PATH = APP_DIR / "logs" / "api.log"

REQUEST_COUNTER = Counter("heart_disease_requests_total", "Total requests received by the API")
REQUEST_LATENCY = Histogram("heart_disease_request_latency_seconds", "API request latency in seconds")

logger = logging.getLogger("heart_disease_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_training(DATA_PATH, MODEL_PATH.parent)


def load_model():
    ensure_model_exists()
    return joblib.load(MODEL_PATH)


@app.middleware("http")
async def log_and_measure_requests(request, call_next):
    start = time.perf_counter()
    REQUEST_COUNTER.inc()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    logger.info("%s %s %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/")
def root() -> dict:
    return {"message": "Heart disease prediction API is running."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PatientInput) -> dict:
    model = load_model()
    record = payload.model_dump()
    result = predict_record(model, record)
    return {"features": {name: record[name] for name in FEATURE_COLUMNS}, **result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("heart_disease.api:app", host="0.0.0.0", port=8000)
