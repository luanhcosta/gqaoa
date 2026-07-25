import logging
from typing import Optional

import pennylane as qml
import mlflow
from pennylane import numpy as pnp

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
    device_name: str = "lightning.gpu",
    run_name: str = "pennylane_gd",
    checkpoint_in: Optional[str] = None,
    checkpoint_out: Optional[str] = None,
) -> StrategyResult:
    """Classical baseline: PennyLane GradientDescentOptimizer + SPSA.

    checkpoint_in/checkpoint_out are accepted for interface uniformity with
    gqaoa_strategy but are no-ops here — there is no neural net state to persist.
    `training.optimizer_lr` is used as the gradient-descent stepsize.
    """
    if checkpoint_in is not None or checkpoint_out is not None:
        logging.info("gradient_descent_strategy has no model state to checkpoint; ignoring checkpoint_in/out")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "optimizer_stepsize": training.optimizer_lr,
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
        optimizer = qml.GradientDescentOptimizer(stepsize=training.optimizer_lr)
        qnode_cost_function = qml.QNode(qaoa.cost_function, device, diff_method="spsa")
        params = pnp.random.uniform(0, 2*pnp.pi, 2*training.depth, requires_grad=True)

        exp_val = None
        for i in range(1, training.limit_qpu_call+1):
            params, exp_val = optimizer.step_and_cost(qnode_cost_function, params)
            logging.info(f'epoch {i} — exp_val: {exp_val}')
            mlflow.log_metric("exp_val", float(exp_val), step=i)

        device_probs = get_device(device_name, wires=qaoa.n_assets, shots=training.shots_probs_result)
        qnode_probability = qml.QNode(qaoa.probability_circuit, device_probs)
        probs = qnode_probability(params)

        output: StrategyResult = {
                "params": params,
                "probs": probs,
                "final_exp_val": exp_val,
                "cov_matrix": cov_matrix,
                "expected_value": expected_value,
                "lambda_sdp": lam,
               }
        mlflow.log_metric("final_exp_val", float(exp_val))
        return output
