from typing import Optional, Protocol, TypedDict

import numpy as np

from gqaoa.config import ProblemConfig, TrainingConfig


class StrategyResult(TypedDict):
    params: np.ndarray
    probs: np.ndarray
    final_exp_val: float
    cov_matrix: object
    expected_value: object
    lambda_sdp: Optional[float]


class OptimizerStrategy(Protocol):
    """Common shape shared by gqaoa_strategy, gradient_descent_strategy and scipy_strategy.

    checkpoint_in/checkpoint_out are part of the interface for all three so that
    experiments/bracket.py and experiments/stability.py can call any strategy
    uniformly through a registry, even though the classical strategies (no
    neural net) simply no-op them.
    """

    def run_job(
        self,
        problem: ProblemConfig,
        training: TrainingConfig,
        *,
        device_name: str = "lightning.gpu",
        run_name: str = "run",
        checkpoint_in: Optional[str] = None,
        checkpoint_out: Optional[str] = None,
    ) -> StrategyResult: ...
