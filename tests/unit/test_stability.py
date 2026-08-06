from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.experiments.stability import run_stability
from gqaoa.strategies import scipy_strategy


def test_run_stability_threads_strategy_kwargs_to_scipy(monkeypatch):
    captured_methods = []

    def fake_run_job(problem, training, *, minimize_method="COBYLA", device_name, run_name,
                      checkpoint_in=None, checkpoint_out=None):
        captured_methods.append(minimize_method)
        return {"final_exp_val": 0.0}

    monkeypatch.setattr(scipy_strategy, "run_job", fake_run_job)

    run_stability(
        strategy="scipy",
        base_config=BEST_KNOWN_CONFIG,
        n_runs=2,
        experiment_name="pytest-default",
        device_name="default.qubit",
        strategy_kwargs={"minimize_method": "Nelder-Mead"},
    )

    assert captured_methods == ["Nelder-Mead", "Nelder-Mead"]


def test_run_stability_defaults_to_no_strategy_kwargs(monkeypatch):
    captured_kwargs = []

    def fake_run_job(problem, training, *, device_name, run_name, **kwargs):
        captured_kwargs.append(kwargs)
        return {"final_exp_val": 0.0}

    monkeypatch.setattr(scipy_strategy, "run_job", fake_run_job)

    run_stability(
        strategy="scipy",
        base_config=BEST_KNOWN_CONFIG,
        n_runs=1,
        experiment_name="pytest-default",
        device_name="default.qubit",
    )

    assert captured_kwargs == [{}]
