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
from gqaoa.reporting.optimality import compare_runs_to_optimal, derive_bitstring, try_load_brute_force
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
    probs_by_run = []
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        result = call(base_config, device_name, f"{run_name_prefix}_{i+1}", **strategy_kwargs)
        e = float(result["final_exp_val"])
        energies.append(e)
        probs_by_run.append(result.get("probs"))
        print(f"  energy_min = {e:.6f}")

    stats = report_stats(
        energies,
        f"Stability report — {strategy}, {n_runs} runs, limit_qpu_call={base_config.training.limit_qpu_call}",
    )

    brute_force_result = try_load_brute_force(base_config.problem.problem_id)
    optimality = None
    if brute_force_result is not None:
        bitstrings = [derive_bitstring(probs) for probs in probs_by_run]
        optimality = compare_runs_to_optimal(bitstrings, energies, brute_force_result)
        print(f"\n{'='*45}")
        print(f"Optimality comparison — {strategy}, {n_runs} runs")
        print(f"{'='*45}")
        for k, v in optimality.items():
            print(f"  {k:28s}: {v}")

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
        if optimality is not None:
            mlflow.log_params({
                # MLflow params must be scalar/string; a tied mode is
                # serialized as a comma-separated string.
                "solution_mode": ",".join(optimality["solution_mode"]),
                "solution_mode_count": optimality["solution_mode_count"],
                "solution_mode_matches_optimal": optimality["solution_mode_matches_optimal"],
                "optimal_bitstring": optimality["optimal_bitstring"],
            })
            mlflow.log_metric("mean_energy_gap", optimality["mean_energy_gap"])
            mlflow.log_metric("optimal_energy_reached_rate", optimality["optimal_energy_reached_rate"])

    return energies, stats
