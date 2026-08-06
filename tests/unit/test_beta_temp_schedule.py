import pytest

from gqaoa.config import TrainingConfig
from gqaoa.strategies.common import beta_temp_schedule


def _training(**overrides):
    base = dict(beta_temp=1.0, beta_temp_max=4.0, beta_temp_anneal_frac=0.8, limit_qpu_call=100)
    base.update(overrides)
    return TrainingConfig(**base)


def test_no_annealing_when_beta_temp_max_is_none():
    training = _training(beta_temp_max=None)
    for count in (0, 1, 50, 100, 1000):
        assert beta_temp_schedule(training, count) == training.beta_temp


def test_annealing_starts_at_beta_temp_max():
    training = _training()
    assert beta_temp_schedule(training, 0) == training.beta_temp_max


def test_annealing_reaches_beta_temp_at_anneal_fraction_and_stays_there():
    training = _training()
    anneal_qpu = training.beta_temp_anneal_frac * training.limit_qpu_call
    assert beta_temp_schedule(training, int(anneal_qpu)) == pytest.approx(training.beta_temp)
    assert beta_temp_schedule(training, training.limit_qpu_call) == pytest.approx(training.beta_temp)
    assert beta_temp_schedule(training, training.limit_qpu_call * 10) == pytest.approx(training.beta_temp)
