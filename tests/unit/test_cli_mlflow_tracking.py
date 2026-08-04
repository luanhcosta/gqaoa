import mlflow

from gqaoa.cli import run_gqaoa


def test_run_gqaoa_main_forces_correct_tracking_uri(monkeypatch, tmp_path):
    """Regression test: run_gqaoa.py used to never call init_mlflow(), so its
    tracking URI was whatever happened to be ambient for the process (in
    practice, an uncontrolled sqlite:///<cwd>/mlflow.db) instead of the shared
    artifacts/mlflow.db every other CLI uses.
    """
    # Simulate an ambient tracking URI that differs from artifacts/mlflow.db —
    # main() must override this via init_mlflow(), not trust whatever was set.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/some-other-arbitrary-path.db")

    fake_result = {
        "params": None, "probs": None, "final_exp_val": -0.1,
        "cov_matrix": None, "expected_value": None, "lambda_sdp": None,
    }
    monkeypatch.setattr(run_gqaoa.gqaoa_strategy, "run_job", lambda *a, **k: fake_result)

    run_gqaoa.main(["--limit-epochs", "1", "--limit-qpu-call", "1"])

    import gqaoa.tracking.mlflow_utils as mlflow_utils
    assert mlflow.get_tracking_uri() == mlflow_utils.MLFLOW_TRACKING_URI
