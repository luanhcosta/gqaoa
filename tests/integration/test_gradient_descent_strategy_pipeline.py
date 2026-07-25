import pytest

from gqaoa.config import ProblemConfig, TrainingConfig
from gqaoa.strategies import gradient_descent_strategy


def test_gradient_descent_strategy_run_job_end_to_end(device_name):
    problem = ProblemConfig(q=0.3, B=5, lamb=0, initial_state="dicke_state", mixture_layer="xy", sdp=False)
    training = TrainingConfig(depth=1, limit_qpu_call=2, optimizer_lr=0.1, shots_probs_result=20)

    result = gradient_descent_strategy.run_job(
        problem, training, device_name=device_name, run_name="test-gd",
    )

    assert set(result.keys()) == {
        "params", "probs", "final_exp_val", "cov_matrix", "expected_value", "lambda_sdp",
    }
    assert result["probs"].sum() == pytest.approx(1.0, abs=1e-6)
    assert result["cov_matrix"].shape == (10, 10)
