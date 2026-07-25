from typing import Optional

import mlflow

from gqaoa.paths import MLFLOW_TRACKING_URI, ARTIFACTS_DIR


def init_mlflow(experiment_name: str, tracking_uri: Optional[str] = None) -> None:
    """Point MLflow at the shared artifacts/mlflow.db and select an experiment.

    Replaces the `mlflow.set_tracking_uri("sqlite:///mlflow.db")` +
    `mlflow.set_experiment(...)` boilerplate that used to be repeated at the
    top of every experiment script's __main__ block.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri or MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
