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
