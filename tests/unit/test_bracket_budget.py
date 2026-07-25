from gqaoa.experiments import bracket as bracket_module


def test_budget_constants_sum_to_1000():
    total = (
        bracket_module.N_PHASE1 * bracket_module.QPU_PHASE1
        + bracket_module.TOP_K * bracket_module.QPU_PHASE2
        + bracket_module.QPU_PHASE3
    )
    assert total == 1000
    assert bracket_module.TOTAL_QPU_CALLS_PER_BRACKET == 1000


def test_n_repetitions_scales_run_count_without_changing_per_run_budget(monkeypatch, tmp_path):
    calls = []

    def fake_run_job(problem, training, model, *, device_name, run_name, checkpoint_in=None, checkpoint_out=None):
        calls.append((run_name, training.limit_qpu_call))
        return {
            "final_exp_val": -0.01 * len(calls),
            "params": None, "probs": None,
            "cov_matrix": None, "expected_value": None, "lambda_sdp": None,
        }

    monkeypatch.setattr(bracket_module.gqaoa_strategy, "run_job", fake_run_job)
    monkeypatch.setattr(bracket_module, "CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setattr(bracket_module, "init_mlflow", lambda experiment_name: None)

    runs_per_bracket = bracket_module.N_PHASE1 + bracket_module.TOP_K + 1

    bracket_module.run_bracket(n_repetitions=1, cleanup_checkpoints=True, device_name="default.qubit")
    assert len(calls) == runs_per_bracket

    calls.clear()
    bracket_module.run_bracket(n_repetitions=3, cleanup_checkpoints=True, device_name="default.qubit")
    assert len(calls) == runs_per_bracket * 3

    # per-run QPU budgets are the same regardless of how many times the bracket repeats
    qpu_budgets = {qpu for _, qpu in calls}
    assert qpu_budgets == {bracket_module.QPU_PHASE1, bracket_module.QPU_PHASE2, bracket_module.QPU_PHASE3}
