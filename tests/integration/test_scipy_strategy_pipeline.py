import pytest

from gqaoa.config import ProblemConfig, TrainingConfig
from gqaoa.strategies import scipy_strategy


def test_scipy_strategy_run_job_end_to_end(device_name):
    problem = ProblemConfig(q=0.3, B=5, lamb=0, initial_state="dicke_state", mixture_layer="xy", sdp=False)
    training = TrainingConfig(depth=1, limit_qpu_call=5, shots_probs_result=20)

    result = scipy_strategy.run_job(
        problem, training, minimize_method="COBYLA", device_name=device_name, run_name="test-scipy",
    )

    assert set(result.keys()) == {
        "params", "probs", "final_exp_val", "cov_matrix", "expected_value", "lambda_sdp",
    }
    assert result["probs"].sum() == pytest.approx(1.0, abs=1e-6)
    assert result["cov_matrix"].shape == (10, 10)


def test_scipy_strategy_survives_gradient_method_exceeding_qpu_budget(device_name):
    # Gradient-based methods (BFGS here) estimate the gradient by finite
    # differences, spending several function evaluations per solver iteration —
    # with a small limit_qpu_call this reliably overruns the budget before
    # scipy's own maxiter (an iteration count, not an eval count) stops it.
    # Regression test for the crash this used to cause (unhandled exception
    # propagating out of minimize()); must complete and return the best point
    # seen so far instead.
    problem = ProblemConfig(q=0.3, B=5, lamb=0, initial_state="dicke_state", mixture_layer="xy", sdp=False)
    training = TrainingConfig(depth=1, limit_qpu_call=3, shots_probs_result=20)

    result = scipy_strategy.run_job(
        problem, training, minimize_method="BFGS", device_name=device_name, run_name="test-scipy-bfgs",
    )

    assert set(result.keys()) == {
        "params", "probs", "final_exp_val", "cov_matrix", "expected_value", "lambda_sdp",
    }
    assert result["probs"].sum() == pytest.approx(1.0, abs=1e-6)
