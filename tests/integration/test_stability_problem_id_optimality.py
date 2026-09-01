import mlflow
import pandas as pd
import pytest

from gqaoa.config import BestKnownConfig, ModelConfig, ProblemConfig, TrainingConfig
from gqaoa.domain.brute_force import brute_force_search
from gqaoa.experiments.stability import run_stability
from gqaoa.problem import brute_force_store, store
from gqaoa.problem.brute_force_store import save_brute_force_result
from gqaoa.problem.store import save_problem
from gqaoa.problem.synthetic import generate_synthetic_problem


@pytest.fixture(autouse=True)
def _isolated_problems_dir(tmp_path, monkeypatch):
    problems_dir = tmp_path / "problems"
    monkeypatch.setattr(store, "PROBLEMS_DIR", problems_dir)
    monkeypatch.setattr(brute_force_store, "PROBLEMS_DIR", problems_dir)
    yield


def test_run_stability_with_problem_id_reports_and_logs_optimality_comparison(device_name):
    instance = generate_synthetic_problem(n_assets=4, seed=11, n_trading_days=60, topology="ring")
    save_problem(instance)

    q, B, lamb = 0.3, 2, 0.0
    brute_force_result = brute_force_search(
        expected_value=instance.expected_value,
        cov_matrix=instance.cov_matrix,
        q=q, B=B, lamb=lamb,
        edges_hc=instance.edges_hc,
        n_assets=instance.n_assets,
        search_mode="full",
    )
    save_brute_force_result(instance.problem_id, q=q, B=B, lamb=lamb, result=brute_force_result)

    config = BestKnownConfig(
        problem=ProblemConfig(
            q=q, B=B, lamb=lamb,
            initial_state="dicke_state", mixture_layer="xy",
            edges_hc=list(instance.edges_hc), edges_hb=list(instance.edges_hb),
            sdp=False, problem_id=instance.problem_id,
        ),
        model=ModelConfig(),
        training=TrainingConfig(depth=1, limit_qpu_call=5, shots_probs_result=50),
    )

    experiment_name = "test-gqaoa-optimality"
    energies, stats = run_stability(
        strategy="scipy",
        base_config=config,
        n_runs=2,
        experiment_name=experiment_name,
        device_name=device_name,
        run_name_prefix="test-optimality",
    )

    assert len(energies) == 2
    assert set(stats.keys()) >= {"min", "max", "mean", "median"}

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.mlflow.runName = 'stability_summary'",
    )
    assert len(runs) == 1
    run = runs.iloc[0]

    assert run["params.optimal_bitstring"] == brute_force_result.optimal_bitstring
    assert run["params.solution_mode"] != ""
    assert int(run["params.solution_mode_count"]) >= 1
    assert not pd.isna(run["metrics.mean_energy_gap"])
    assert 0.0 <= run["metrics.optimal_energy_reached_rate"] <= 1.0
