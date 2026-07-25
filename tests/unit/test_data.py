import inspect

from gqaoa.domain import data
from gqaoa.domain.data import f_return_cov


def test_returns_expected_shapes():
    expected_value, cov_matrix = f_return_cov()
    assert len(expected_value) == 10
    assert cov_matrix.shape == (10, 10)


def test_covariance_matrix_is_symmetric():
    _, cov_matrix = f_return_cov()
    assert (cov_matrix.values == cov_matrix.values.T).all()


def test_no_duplicate_f_return_cov_definition():
    # Regression test for the old src/data_qubo.py bug where a second
    # `def f_return_cov():` silently shadowed the first.
    source = inspect.getsource(data)
    assert source.count("def f_return_cov(") == 1
