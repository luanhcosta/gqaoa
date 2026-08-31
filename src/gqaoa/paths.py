import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("GQAOA_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))

MLFLOW_DB_PATH = ARTIFACTS_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

OPTUNA_DB_PATH = ARTIFACTS_DIR / "optuna.db"
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB_PATH}"

CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

PROBLEMS_DIR = ARTIFACTS_DIR / "problems"
