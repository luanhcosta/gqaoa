import logging
from typing import Optional

import numpy as np
import pennylane as qml
import mlflow
from scipy.optimize import minimize

from gqaoa.config import ProblemConfig, TrainingConfig
from gqaoa.domain.device import get_device
from gqaoa.domain.qaoa import QAOA
from gqaoa.strategies.base import StrategyResult
from gqaoa.strategies.common import build_problem

logging.basicConfig(level=logging.INFO)


def run_job(
    problem: ProblemConfig,
    training: TrainingConfig,
    *,
    minimize_method: str = "COBYLA",
    device_name: str = "lightning.gpu",
    run_name: str = "scipy_minimize",
    checkpoint_in: Optional[str] = None,
    checkpoint_out: Optional[str] = None,
) -> StrategyResult:
    """Classical baseline: scipy.optimize.minimize over the QAOA cost function.

    checkpoint_in/checkpoint_out are accepted for interface uniformity with
    gqaoa_strategy but are no-ops here — there is no neural net state to persist.
    """
    if checkpoint_in is not None or checkpoint_out is not None:
        logging.info("scipy_strategy has no model state to checkpoint; ignoring checkpoint_in/out")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "minimize_method": minimize_method,
            "q": problem.q,
            "B": problem.B,
            "lamb": problem.lamb,
            "initial_state": problem.initial_state,
            "mixture_layer": problem.mixture_layer,
            "sdp": problem.sdp,
            "depth": training.depth,
            "limit_qpu_call": training.limit_qpu_call,
        })

        expected_value, cov_matrix, lam = build_problem(problem)

        qaoa = QAOA(
            expected_value,
            cov_matrix,
            problem.q,
            problem.B,
            problem.lamb,
            initial_state=problem.initial_state,
            mixture_layer=problem.mixture_layer,
            edges_hc=problem.edges_hc,
            edges_hb=problem.edges_hb,
        )

        device = get_device(device_name, wires=qaoa.n_assets)
        qnode_cost_function = qml.QNode(qaoa.cost_function, device)

        def objective(params):
            if qaoa.count_qpu_call > training.limit_qpu_call:
                raise Exception(f"Limit exceeded for calls on QPU. limit_qpu_call={training.limit_qpu_call}")
            exp_val = qnode_cost_function(params)
            mlflow.log_metric("exp_val", float(exp_val), step=qaoa.count_qpu_call)
            return exp_val

        initial_params = np.random.uniform(0, 2*np.pi, 2*training.depth)
        opt_result = minimize(
            objective, initial_params, method=minimize_method,
            options={'maxiter': training.limit_qpu_call, 'maxfev': training.limit_qpu_call},
        )

        device_probs = get_device(device_name, wires=qaoa.n_assets, shots=training.shots_probs_result)
        qnode_probability = qml.QNode(qaoa.probability_circuit, device_probs)
        probs = qnode_probability(opt_result.x)

        output: StrategyResult = {
                "params": opt_result.x,
                "probs": probs,
                "final_exp_val": opt_result.fun,
                "cov_matrix": cov_matrix,
                "expected_value": expected_value,
                "lambda_sdp": lam,
               }
        mlflow.log_metric("final_exp_val", float(opt_result.fun))
        return output
