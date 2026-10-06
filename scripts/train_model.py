from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from heart_disease.data_pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heart disease models and log to MLflow.")
    parser.add_argument("--data", type=str, default="data/raw/heart+disease/processed.cleveland.data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / args.data
    artifact_dir = repo_root / args.artifact_dir
    run_training(data_path, artifact_dir)
    print(f"Artifacts saved to {artifact_dir}")


if __name__ == "__main__":
    main()
