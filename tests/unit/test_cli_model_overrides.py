import argparse

from gqaoa.cli import run_bracket, run_hpo, run_stability_bracket, run_stability_check
from gqaoa.cli._common import add_model_override_args, apply_model_overrides
from gqaoa.config import BEST_KNOWN_CONFIG


def test_apply_model_overrides_only_changes_given_fields():
    parser = argparse.ArgumentParser()
    add_model_override_args(parser)
    args = parser.parse_args(["--n-layer", "1", "--vocab-size", "4"])

    config = apply_model_overrides(BEST_KNOWN_CONFIG, args)

    assert config.model.n_layer == 1
    assert config.model.vocab_size == 4
    assert config.model.n_embd == BEST_KNOWN_CONFIG.model.n_embd
    assert config.model.n_head == BEST_KNOWN_CONFIG.model.n_head
    # original config is untouched (dataclasses.replace returns a copy)
    assert BEST_KNOWN_CONFIG.model.n_layer != 1


def test_apply_model_overrides_no_flags_keeps_config_unchanged():
    parser = argparse.ArgumentParser()
    add_model_override_args(parser)
    args = parser.parse_args([])

    config = apply_model_overrides(BEST_KNOWN_CONFIG, args)

    assert config.model == BEST_KNOWN_CONFIG.model


def test_stability_check_cli_exposes_model_override_flags():
    args = run_stability_check.build_arg_parser().parse_args(
        ["--n-layer", "1", "--vocab-size", "4", "--n-runs", "2"]
    )
    assert args.n_layer == 1
    assert args.vocab_size == 4
    assert args.n_embd is None
    assert args.n_head is None


def test_bracket_cli_threads_model_override_into_run_bracket(monkeypatch):
    captured = {}

    def fake_run_bracket(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_bracket, "run_bracket", fake_run_bracket)

    run_bracket.main(["--device-name", "default.qubit", "--n-layer", "1", "--vocab-size", "4"])

    assert captured["base_config"].model.n_layer == 1
    assert captured["base_config"].model.vocab_size == 4
    assert captured["device_name"] == "default.qubit"


def test_stability_bracket_cli_threads_model_override_into_run_bracket(monkeypatch):
    captured = {}

    def fake_run_bracket(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_stability_bracket, "run_bracket", fake_run_bracket)

    run_stability_bracket.main(
        ["--device-name", "default.qubit", "--n-repetitions", "2", "--n-layer", "1", "--vocab-size", "4"]
    )

    assert captured["n_repetitions"] == 2
    assert captured["base_config"].model.n_layer == 1
    assert captured["base_config"].model.vocab_size == 4


def test_hpo_cli_exposes_and_threads_limit_flags(monkeypatch):
    captured = {}

    def fake_run_hpo_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_hpo, "run_hpo_main", fake_run_hpo_main)

    run_hpo.main(["--n-trials", "2", "--limit-epochs", "5", "--limit-qpu-call", "10", "--device-name", "default.qubit"])

    assert captured == {
        "n_trials": 2,
        "device_name": "default.qubit",
        "limit_epochs": 5,
        "limit_qpu_call": 10,
    }


def test_hpo_cli_default_limits_match_objective_defaults():
    args = run_hpo.build_arg_parser().parse_args([])
    assert args.limit_epochs == 900
    assert args.limit_qpu_call == 200


def test_stability_check_cli_default_uses_annealing():
    args = run_stability_check.build_arg_parser().parse_args([])
    assert args.no_annealing is False


def test_stability_check_cli_default_threads_best_known_config(monkeypatch):
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_stability_check, "run_stability", fake_run_stability)
    run_stability_check.main(["--limit-qpu-call", "50"])

    assert captured["base_config"].training.beta_temp_max == 4.0
    assert captured["run_name_prefix"] == "stability_anneal_50"


def test_stability_check_cli_no_annealing_flag_threads_no_anneal_config(monkeypatch):
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_stability_check, "run_stability", fake_run_stability)
    run_stability_check.main(["--no-annealing", "--n-runs", "1", "--limit-qpu-call", "50"])

    assert captured["base_config"].training.beta_temp_max is None
    assert captured["run_name_prefix"] == "stability_no_anneal_50"
    assert captured["n_runs"] == 1
