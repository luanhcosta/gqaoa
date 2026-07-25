import pytest

pytestmark = pytest.mark.slow


def test_one_real_optuna_trial_end_to_end():
    optuna = pytest.importorskip("optuna")
    from gqaoa.experiments.hpo import objective

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial, device_name="default.qubit", limit_epochs=2, limit_qpu_call=3,
        ),
        n_trials=1,
    )

    assert len(study.trials) == 1
    assert study.best_trial.value is not None
