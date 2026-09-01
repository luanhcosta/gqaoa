import argparse
import dataclasses

from gqaoa.cli import (
    run_benchmark_gd,
    run_benchmark_scipy,
    run_gqaoa,
    run_hpo,
    run_stability_check,
)
from gqaoa.cli._common import add_sdp_override_args, apply_sdp_override
from gqaoa.config import BEST_KNOWN_CONFIG


def test_apply_sdp_override_no_flag_keeps_config_unchanged():
    parser = argparse.ArgumentParser()
    add_sdp_override_args(parser)
    args = parser.parse_args([])

    config = apply_sdp_override(BEST_KNOWN_CONFIG, args)

    assert config is BEST_KNOWN_CONFIG
    assert config.problem.sdp is True


def test_apply_sdp_override_no_sdp_flag_disables_sdp():
    parser = argparse.ArgumentParser()
    add_sdp_override_args(parser)
    args = parser.parse_args(["--no-sdp"])

    config = apply_sdp_override(BEST_KNOWN_CONFIG, args)

    assert config.problem.sdp is False
    # only the sdp field changed
    assert config.problem == dataclasses.replace(BEST_KNOWN_CONFIG.problem, sdp=False)
    # original config is untouched (dataclasses.replace returns a copy)
    assert BEST_KNOWN_CONFIG.problem.sdp is True


def test_stability_check_cli_no_sdp_flag_threads_into_run_stability(monkeypatch):
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_stability_check, "run_stability", fake_run_stability)

    run_stability_check.main(["--no-sdp", "--n-runs", "1", "--limit-qpu-call", "50"])

    assert captured["base_config"].problem.sdp is False


def test_benchmark_gd_cli_no_sdp_flag_threads_into_run_stability(monkeypatch):
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_benchmark_gd, "run_stability", fake_run_stability)

    run_benchmark_gd.main(["--no-sdp", "--n-runs", "1"])

    assert captured["base_config"].problem.sdp is False


def test_benchmark_scipy_cli_no_sdp_flag_threads_into_run_stability(monkeypatch):
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_benchmark_scipy, "run_stability", fake_run_stability)

    run_benchmark_scipy.main(["--no-sdp", "--n-runs", "1"])

    assert captured["base_config"].problem.sdp is False


def test_gqaoa_cli_no_sdp_flag_disables_sdp_in_problem_config(monkeypatch):
    captured = {}

    def fake_run_job(problem, training, model, **kwargs):
        captured["problem"] = problem
        return {}

    monkeypatch.setattr(run_gqaoa.gqaoa_strategy, "run_job", fake_run_job)
    monkeypatch.setattr(run_gqaoa, "init_mlflow", lambda *a, **k: None)

    run_gqaoa.main(["--no-sdp", "--device-name", "default.qubit"])

    assert captured["problem"].sdp is False


def test_gqaoa_cli_default_keeps_sdp_enabled(monkeypatch):
    captured = {}

    def fake_run_job(problem, training, model, **kwargs):
        captured["problem"] = problem
        return {}

    monkeypatch.setattr(run_gqaoa.gqaoa_strategy, "run_job", fake_run_job)
    monkeypatch.setattr(run_gqaoa, "init_mlflow", lambda *a, **k: None)

    run_gqaoa.main(["--device-name", "default.qubit"])

    assert captured["problem"].sdp is True


def test_hpo_cli_no_sdp_flag_threads_sdp_false(monkeypatch):
    captured = {}

    def fake_run_hpo_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_hpo, "run_hpo_main", fake_run_hpo_main)

    run_hpo.main(["--no-sdp", "--n-trials", "1"])

    assert captured["sdp"] is False
