import pytest

from gqaoa.config import ModelConfig, ProblemConfig, TrainingConfig
from gqaoa.strategies import gqaoa_strategy


def test_gqaoa_strategy_run_job_end_to_end(device_name):
    problem = ProblemConfig(q=0.3, B=5, lamb=0, initial_state="dicke_state", mixture_layer="xy", sdp=False)
    training = TrainingConfig(depth=1, limit_epochs=2, limit_qpu_call=3, optimizer_lr=0.01, shots_probs_result=50)
    model = ModelConfig(vocab_size=4, n_embd=32, n_layer=1, n_head=1)

    result = gqaoa_strategy.run_job(
        problem, training, model, device_name=device_name, run_name="test-gqaoa",
    )

    assert set(result.keys()) == {
        "params", "probs", "final_exp_val", "cov_matrix", "expected_value", "lambda_sdp",
    }
    assert result["probs"].sum() == pytest.approx(1.0, abs=1e-6)
    assert result["cov_matrix"].shape == (10, 10)
    assert len(result["expected_value"]) == 10
    assert result["lambda_sdp"] is None


def test_gqaoa_strategy_survives_transient_mlflow_logging_failure(device_name, monkeypatch):
    """Regression test: a transient MLflow/SQLite error (e.g. lock contention
    with a concurrently running `mlflow ui`, which is exactly what the README
    tells users to run alongside training) used to crash the whole run and
    discard every epoch completed so far. The per-epoch metrics block is now
    wrapped in try/except; this simulates every per-epoch log_metric call
    failing and asserts the run still completes normally.
    """
    import mlflow

    real_log_metric = mlflow.log_metric

    def flaky_log_metric(key, value, step=None, **kwargs):
        if step is not None:
            # only the per-epoch calls pass step=i; the one-shot final
            # "final_exp_val" log_metric call (outside the try/except by
            # design) does not, so it is left free to succeed normally.
            raise RuntimeError("simulated transient MLflow/SQLite failure")
        return real_log_metric(key, value, step=step, **kwargs)

    monkeypatch.setattr(mlflow, "log_metric", flaky_log_metric)

    problem = ProblemConfig(q=0.3, B=5, lamb=0, initial_state="dicke_state", mixture_layer="xy", sdp=False)
    training = TrainingConfig(depth=1, limit_epochs=2, limit_qpu_call=3, optimizer_lr=0.01, shots_probs_result=50)
    model = ModelConfig(vocab_size=4, n_embd=32, n_layer=1, n_head=1)

    result = gqaoa_strategy.run_job(
        problem, training, model, device_name=device_name, run_name="test-gqaoa-flaky-mlflow",
    )

    assert set(result.keys()) == {
        "params", "probs", "final_exp_val", "cov_matrix", "expected_value", "lambda_sdp",
    }
    assert result["probs"].sum() == pytest.approx(1.0, abs=1e-6)
