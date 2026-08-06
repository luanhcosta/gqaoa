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


class _QpuBudgetExceeded(Exception):
    """Raised by objective() once limit_qpu_call is spent, to unwind out of minimize()."""


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

    Derivative-free methods (COBYLA, Nelder-Mead, Powell) keep their own function-eval
    count within `maxiter`/`maxfev`, so they stop on their own by limit_qpu_call.
    Gradient-based methods (BFGS, CG, L-BFGS-B, SLSQP, ...) estimate the gradient by
    finite differences, spending several evaluations per solver iteration, and can
    blow past limit_qpu_call before `maxiter` (an iteration count, not an eval count)
    ever triggers — in that case objective() aborts the solver early and the best
    point/energy seen so far is returned instead of letting the budget overrun crash
    the run.
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

        initial_params = np.random.uniform(0, 2*np.pi, 2*training.depth)
        best = {"params": initial_params.copy(), "exp_val": np.inf}

        def objective(params):
            if qaoa.count_qpu_call > training.limit_qpu_call:
                raise _QpuBudgetExceeded()
            exp_val = qnode_cost_function(params)
            mlflow.log_metric("exp_val", float(exp_val), step=qaoa.count_qpu_call)
            if exp_val < best["exp_val"]:
                best["exp_val"] = float(exp_val)
                best["params"] = np.array(params, copy=True)
            return exp_val

        try:
            opt_result = minimize(
                objective, initial_params, method=minimize_method,
                options={'maxiter': training.limit_qpu_call, 'maxfev': training.limit_qpu_call},
            )
            final_params, final_exp_val = opt_result.x, float(opt_result.fun)
        except _QpuBudgetExceeded:
            logging.info(
                f"minimize_method={minimize_method} exceeded limit_qpu_call="
                f"{training.limit_qpu_call} mid-iteration (its own maxiter/maxfev "
                f"counts iterations, not evaluations); using best point seen so far "
                f"(exp_val={best['exp_val']:.6f})"
            )
            mlflow.log_param("qpu_budget_exceeded", True)
            final_params, final_exp_val = best["params"], best["exp_val"]

        device_probs = get_device(device_name, wires=qaoa.n_assets, shots=training.shots_probs_result)
        qnode_probability = qml.QNode(qaoa.probability_circuit, device_probs)
        probs = qnode_probability(final_params)

        output: StrategyResult = {
                "params": final_params,
                "probs": probs,
                "final_exp_val": final_exp_val,
                "cov_matrix": cov_matrix,
                "expected_value": expected_value,
                "lambda_sdp": lam,
               }
        mlflow.log_metric("final_exp_val", final_exp_val)
        return output
