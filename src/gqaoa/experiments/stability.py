"""Repeated-run stability analysis, generalized over optimizer strategy.

Unifies the old stability_check.py (strategy="gqaoa") and benchmark_gd.py
(strategy="gradient_descent") into one function, and fixes the bug where
benchmark_gd.py logged to the same MLflow experiment name as
stability_check.py — callers must now pass an explicit experiment_name.
"""
import dataclasses
from typing import Literal, Optional

import mlflow

from gqaoa.config import BestKnownConfig
from gqaoa.reporting.stats import report_stats
from gqaoa.strategies import gqaoa_strategy, gradient_descent_strategy, scipy_strategy
from gqaoa.tracking.mlflow_utils import init_mlflow

Strategy = Literal["gqaoa", "gradient_descent", "scipy"]


def _call_gqaoa(config: BestKnownConfig, device_name, run_name, **strategy_kwargs):
    return gqaoa_strategy.run_job(config.problem, config.training, config.model,
                                   device_name=device_name, run_name=run_name, **strategy_kwargs)


def _call_gradient_descent(config: BestKnownConfig, device_name, run_name, **strategy_kwargs):
    return gradient_descent_strategy.run_job(config.problem, config.training,
                                              device_name=device_name, run_name=run_name, **strategy_kwargs)


def _call_scipy(config: BestKnownConfig, device_name, run_name, **strategy_kwargs):
    return scipy_strategy.run_job(config.problem, config.training,
                                   device_name=device_name, run_name=run_name, **strategy_kwargs)


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
    strategy_kwargs: Optional[dict] = None,
) -> tuple[list, dict]:
    """Run `strategy` `n_runs` times with identical config and report energy_min statistics.

    `strategy_kwargs` are forwarded verbatim to the strategy's run_job() on every
    run — e.g. {"minimize_method": "Nelder-Mead"} for the scipy strategy, whose
    optimizer algorithm isn't a TrainingConfig field. Also logged as MLflow params
    on the stability_summary run so the algorithm used is visible alongside the stats.
    """
    call = _STRATEGIES[strategy]
    strategy_kwargs = strategy_kwargs or {}
    init_mlflow(experiment_name)

    energies = []
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        result = call(base_config, device_name, f"{run_name_prefix}_{i+1}", **strategy_kwargs)
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
            **strategy_kwargs,
            **dataclasses.asdict(base_config.problem),
            **dataclasses.asdict(base_config.training),
        })
        for k, v in stats.items():
            mlflow.log_metric(k, v)

    return energies, stats
