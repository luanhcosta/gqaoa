import argparse

import pytest

from gqaoa.cli import run_benchmark_gd, run_benchmark_scipy, run_stability_check
from gqaoa.cli._common import add_problem_id_arg, apply_problem_id
from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.problem import store
from gqaoa.problem.synthetic import generate_synthetic_problem
from gqaoa.problem.store import save_problem


@pytest.fixture(autouse=True)
def _isolated_problems_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROBLEMS_DIR", tmp_path / "problems")
    yield


def _persist_synthetic_problem(n_assets=5, seed=1):
    instance = generate_synthetic_problem(n_assets=n_assets, seed=seed, n_trading_days=60)
    save_problem(instance)
    return instance


def test_apply_problem_id_no_flag_keeps_config_unchanged():
    parser = argparse.ArgumentParser()
    add_problem_id_arg(parser)
    args = parser.parse_args([])

    config = apply_problem_id(BEST_KNOWN_CONFIG, args)

    assert config is BEST_KNOWN_CONFIG
    assert config.problem.problem_id is None


def test_apply_problem_id_sets_problem_id_and_topology_from_instance():
    instance = _persist_synthetic_problem(n_assets=5, seed=2)
    parser = argparse.ArgumentParser()
    add_problem_id_arg(parser)
    args = parser.parse_args(["--problem-id", instance.problem_id])

    config = apply_problem_id(BEST_KNOWN_CONFIG, args)

    assert config.problem.problem_id == instance.problem_id
    assert config.problem.edges_hc == list(instance.edges_hc)
    assert config.problem.edges_hb == list(instance.edges_hb)
    # only problem_id/edges_hc/edges_hb changed
    assert config.problem.q == BEST_KNOWN_CONFIG.problem.q
    assert config.problem.sdp == BEST_KNOWN_CONFIG.problem.sdp
    # original config is untouched (dataclasses.replace returns a copy)
    assert BEST_KNOWN_CONFIG.problem.problem_id is None


def test_stability_check_cli_exposes_problem_id_flag():
    args = run_stability_check.build_arg_parser().parse_args([])
    assert args.problem_id is None


def test_stability_check_cli_threads_problem_id_into_run_stability(monkeypatch):
    instance = _persist_synthetic_problem(n_assets=5, seed=3)
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_stability_check, "run_stability", fake_run_stability)

    run_stability_check.main(
        ["--n-runs", "1", "--limit-qpu-call", "50", "--problem-id", instance.problem_id]
    )

    assert captured["base_config"].problem.problem_id == instance.problem_id
    assert captured["base_config"].problem.edges_hc == list(instance.edges_hc)


def test_benchmark_scipy_cli_threads_problem_id_into_run_stability(monkeypatch):
    instance = _persist_synthetic_problem(n_assets=5, seed=4)
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_benchmark_scipy, "run_stability", fake_run_stability)

    run_benchmark_scipy.main(["--n-runs", "1", "--problem-id", instance.problem_id])

    assert captured["base_config"].problem.problem_id == instance.problem_id
    assert captured["base_config"].problem.edges_hc == list(instance.edges_hc)


def test_benchmark_gd_cli_threads_problem_id_into_run_stability(monkeypatch):
    instance = _persist_synthetic_problem(n_assets=5, seed=5)
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_benchmark_gd, "run_stability", fake_run_stability)

    run_benchmark_gd.main(["--n-runs", "1", "--problem-id", instance.problem_id])

    assert captured["base_config"].problem.problem_id == instance.problem_id
    assert captured["base_config"].problem.edges_hc == list(instance.edges_hc)
