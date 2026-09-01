import pandas as pd
import pytest

from gqaoa.config import ProblemConfig
from gqaoa.domain.data import f_return_cov
from gqaoa.problem import store
from gqaoa.problem.synthetic import generate_synthetic_problem
from gqaoa.problem.store import save_problem
from gqaoa.strategies.common import build_problem


@pytest.fixture(autouse=True)
def _isolated_problems_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROBLEMS_DIR", tmp_path / "problems")
    yield


def test_build_problem_without_problem_id_preserves_legacy_behavior():
    problem = ProblemConfig(q=0.3, B=5, lamb=0, sdp=False)

    expected_value, cov_matrix, lam = build_problem(problem)

    ref_expected_value, ref_cov_matrix = f_return_cov()
    assert list(expected_value) == pytest.approx(ref_expected_value)
    assert cov_matrix.to_numpy() == pytest.approx(ref_cov_matrix.to_numpy())
    assert lam is None


def test_build_problem_without_problem_id_still_applies_sdp_over_edges_hc():
    ring_edges = [(0, 4), (4, 1), (1, 7), (7, 2), (2, 8), (8, 5), (5, 9), (9, 6), (6, 3), (3, 0)]
    problem = ProblemConfig(q=0.3, B=5, lamb=0, edges_hc=ring_edges, sdp=True)

    expected_value, cov_matrix, lam = build_problem(problem)

    assert lam is not None
    assert isinstance(cov_matrix, pd.DataFrame)
    assert cov_matrix.shape == (10, 10)


def test_build_problem_with_problem_id_loads_persisted_instance():
    instance = generate_synthetic_problem(n_assets=5, seed=42, n_trading_days=60)
    save_problem(instance)

    problem = ProblemConfig(q=0.3, B=2, lamb=0, sdp=False, problem_id=instance.problem_id)

    expected_value, cov_matrix, lam = build_problem(problem)

    assert list(expected_value) == pytest.approx(instance.expected_value)
    assert cov_matrix.to_numpy() == pytest.approx(instance.cov_matrix.to_numpy())
    assert lam is None


def test_build_problem_with_problem_id_and_sdp_falls_back_to_instance_edges_hc():
    instance = generate_synthetic_problem(n_assets=5, seed=7, n_trading_days=60)
    save_problem(instance)

    # No explicit edges_hc on the config: build_problem must fall back to the
    # loaded instance's own topology for the SDP compression graph, not skip
    # compression or use the (unrelated) fixed RING_TOPOLOGY_EDGES.
    problem = ProblemConfig(q=0.3, B=2, lamb=0, sdp=True, problem_id=instance.problem_id)

    expected_value, cov_matrix, lam = build_problem(problem)

    assert lam is not None
    assert cov_matrix.shape == (5, 5)


def test_build_problem_with_problem_id_prefers_explicit_edges_hc_for_sdp():
    instance = generate_synthetic_problem(n_assets=5, seed=13, n_trading_days=60)
    save_problem(instance)

    explicit_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    problem = ProblemConfig(
        q=0.3, B=2, lamb=0, sdp=True, problem_id=instance.problem_id, edges_hc=explicit_edges,
    )

    expected_value, cov_matrix, lam = build_problem(problem)

    assert lam is not None
    assert cov_matrix.shape == (5, 5)
