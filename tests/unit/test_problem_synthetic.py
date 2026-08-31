import numpy as np
import pytest

from gqaoa.problem.synthetic import generate_synthetic_problem


@pytest.mark.parametrize("n_assets", [5, 10, 50])
def test_shapes(n_assets):
    instance = generate_synthetic_problem(n_assets=n_assets, seed=42)
    assert len(instance.expected_value) == n_assets
    assert instance.cov_matrix.shape == (n_assets, n_assets)
    assert instance.n_assets == n_assets
    assert len(instance.asset_names) == n_assets


@pytest.mark.parametrize("n_assets", [5, 10, 50])
def test_cov_matrix_is_symmetric_and_psd(n_assets):
    instance = generate_synthetic_problem(n_assets=n_assets, seed=7)
    values = instance.cov_matrix.to_numpy()

    assert np.allclose(values, values.T)

    eigenvalues = np.linalg.eigvalsh(values)
    assert eigenvalues.min() >= -1e-8


def test_same_seed_is_fully_deterministic():
    a = generate_synthetic_problem(n_assets=8, seed=123, n_trading_days=100)
    b = generate_synthetic_problem(n_assets=8, seed=123, n_trading_days=100)

    assert a.expected_value == b.expected_value
    assert a.cov_matrix.equals(b.cov_matrix)
    assert a.edges_hc == b.edges_hc
    assert a.provenance == b.provenance
    assert a.problem_id == b.problem_id


def test_different_seed_changes_result():
    a = generate_synthetic_problem(n_assets=8, seed=1, n_trading_days=100)
    b = generate_synthetic_problem(n_assets=8, seed=2, n_trading_days=100)

    assert a.expected_value != b.expected_value
    assert a.problem_id != b.problem_id


def test_seed_none_is_recorded_in_provenance_and_is_reproducible():
    instance = generate_synthetic_problem(n_assets=6, seed=None, n_trading_days=50)
    assert instance.provenance["seed"] is not None
    picked_seed = instance.provenance["seed"]

    replay = generate_synthetic_problem(n_assets=6, seed=picked_seed, n_trading_days=50)
    assert replay.expected_value == instance.expected_value
    assert replay.problem_id == instance.problem_id


def test_default_topology_is_ring():
    instance = generate_synthetic_problem(n_assets=6, seed=1, n_trading_days=50)
    assert len(instance.edges_hc) == 6
    assert instance.provenance["topology"] == "ring"


def test_complete_topology():
    instance = generate_synthetic_problem(n_assets=6, seed=1, n_trading_days=50, topology="complete")
    assert len(instance.edges_hc) == 15


def test_provenance_contains_required_keys():
    instance = generate_synthetic_problem(n_assets=5, seed=1, n_trading_days=50)
    assert set(instance.provenance.keys()) == {
        "n_assets", "seed", "n_trading_days", "topology", "annualization_factor",
    }
    assert instance.source == "synthetic"
    assert instance.schema_version == 1
