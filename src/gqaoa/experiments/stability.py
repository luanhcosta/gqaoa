"""Repeated-run stability analysis, generalized over optimizer strategy.

Unifies the old stability_check.py (strategy="gqaoa") and benchmark_gd.py
(strategy="gradient_descent") into one function, and fixes the bug where
benchmark_gd.py logged to the same MLflow experiment name as
stability_check.py — callers must now pass an explicit experiment_name.
"""
import dataclasses
from typing import Literal

import mlflow

from gqaoa.config import BestKnownConfig
from gqaoa.reporting.stats import report_stats
from gqaoa.strategies import gqaoa_strategy, gradient_descent_strategy, scipy_strategy
from gqaoa.tracking.mlflow_utils import init_mlflow

Strategy = Literal["gqaoa", "gradient_descent", "scipy"]


def _call_gqaoa(config: BestKnownConfig, device_name, run_name):
    return gqaoa_strategy.run_job(config.problem, config.training, config.model,
                                   device_name=device_name, run_name=run_name)


def _call_gradient_descent(config: BestKnownConfig, device_name, run_name):
    return gradient_descent_strategy.run_job(config.problem, config.training,
                                              device_name=device_name, run_name=run_name)


def _call_scipy(config: BestKnownConfig, device_name, run_name):
    return scipy_strategy.run_job(config.problem, config.training,
                                   device_name=device_name, run_name=run_name)


_STRATEGIES = {
    "gqaoa": _call_gqaoa,
    "gradient_descent": _call_gradient_descent,
    "scipy": _call_scipy,
}


def run_stability(
    strategy: Strategy,
    base_config: BestKnownConfig,
    n_runs: int,
    experiment_name: str,
    device_name: str = "lightning.gpu",
    run_name_prefix: str = "stability",
) -> tuple[list, dict]:
    call = _STRATEGIES[strategy]
    init_mlflow(experiment_name)

    energies = []
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        result = call(base_config, device_name, f"{run_name_prefix}_{i+1}")
        e = float(result["final_exp_val"])
        energies.append(e)
        print(f"  energy_min = {e:.6f}")

    stats = report_stats(
        energies,
        f"Stability report — {strategy}, {n_runs} runs, limit_qpu_call={base_config.training.limit_qpu_call}",
    )

    with mlflow.start_run(run_name="stability_summary"):
        mlflow.log_params({
            "strategy": strategy,
            "n_runs": n_runs,
            **dataclasses.asdict(base_config.problem),
            **dataclasses.asdict(base_config.training),
        })
        for k, v in stats.items():
            mlflow.log_metric(k, v)

    return energies, stats
