from __future__ import annotations

import urllib.request
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "data" / "raw" / "heart+disease" / "processed.cleveland.data"


def main() -> None:
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TARGET_PATH.exists():
        print(f"Dataset already exists at {TARGET_PATH}.")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, TARGET_PATH)
    print(f"Dataset saved to {TARGET_PATH}")


if __name__ == "__main__":
    main()
