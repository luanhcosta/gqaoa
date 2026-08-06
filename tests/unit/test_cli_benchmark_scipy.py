from gqaoa.cli import run_benchmark_scipy


def test_benchmark_scipy_cli_defaults():
    args = run_benchmark_scipy.build_arg_parser().parse_args([])
    assert args.n_runs == 10
    assert args.minimize_method == "COBYLA"
    assert args.depth == 5
    assert args.limit_qpu_call == 1000


def test_benchmark_scipy_cli_threads_args_into_run_stability(monkeypatch):
    captured = {}

    def fake_run_stability(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(run_benchmark_scipy, "run_stability", fake_run_stability)

    run_benchmark_scipy.main([
        "--n-runs", "3", "--minimize-method", "Nelder-Mead",
        "--depth", "2", "--limit-qpu-call", "20", "--device-name", "default.qubit",
    ])

    assert captured["strategy"] == "scipy"
    assert captured["n_runs"] == 3
    assert captured["experiment_name"] == "gqaoa-benchmark-scipy"
    assert captured["device_name"] == "default.qubit"
    assert captured["run_name_prefix"] == "benchmark_scipy_Nelder-Mead"
    assert captured["strategy_kwargs"] == {"minimize_method": "Nelder-Mead"}
    assert captured["base_config"].training.depth == 2
    assert captured["base_config"].training.limit_qpu_call == 20
