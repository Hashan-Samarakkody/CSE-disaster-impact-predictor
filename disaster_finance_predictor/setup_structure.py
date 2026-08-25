"""Setup script to initialize the required project directory structure."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/synthetic",
    "src/data_pipeline",
    "src/sampling",
    "src/models",
    "src/training",
    "src/evaluation",
]

FILES = [
    "src/__init__.py",
    "src/data_pipeline/ingester.py",
    "src/data_pipeline/preprocessor.py",
    "src/data_pipeline/feature_eng.py",
    "src/sampling/time_aware_smogn.py",
    "src/models/baselines.py",
    "src/models/tree_ensembles.py",
    "src/models/shallow_mlp.py",
    "src/training/walk_forward.py",
    "src/training/optimizers.py",
    "src/evaluation/metrics.py",
    "src/evaluation/explainability.py",
    "config.yaml",
    "main.py",
]


def initialize() -> None:
    for directory in DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    for file_path in FILES:
        path = ROOT / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


if __name__ == "__main__":
    initialize()
    print("Project structure initialized.")
