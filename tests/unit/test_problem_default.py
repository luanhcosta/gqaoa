from gqaoa.config import DEFAULT_PROBLEM_ID, RING_TOPOLOGY_EDGES
from gqaoa.domain.data import f_return_cov
from gqaoa.problem.default import ASSET_NAMES, default_problem_instance, ensure_default_problem_persisted
from gqaoa.problem.store import load_problem


def test_default_problem_instance_matches_f_return_cov_and_ring_topology():
    expected_value, cov_matrix = f_return_cov()
    instance = default_problem_instance()

    assert instance.problem_id == DEFAULT_PROBLEM_ID == "default-n10-fixed"
    assert instance.source == "default_fixed"
    assert instance.n_assets == 10
    assert instance.asset_names == ASSET_NAMES
    assert instance.expected_value == list(expected_value)
    assert instance.cov_matrix.to_numpy().tolist() == cov_matrix.to_numpy().tolist()
    assert instance.edges_hc == [tuple(e) for e in RING_TOPOLOGY_EDGES]
    assert instance.edges_hb == [tuple(e) for e in RING_TOPOLOGY_EDGES]
    assert instance.schema_version == 1


def test_ensure_default_problem_persisted_bootstraps_on_first_call():
    # Nothing saved yet in the isolated PROBLEMS_DIR (see conftest.py).
    instance = ensure_default_problem_persisted()

    assert instance.problem_id == DEFAULT_PROBLEM_ID
    loaded = load_problem(DEFAULT_PROBLEM_ID)
    assert loaded.expected_value == instance.expected_value
    assert loaded.created_at == instance.created_at


def test_ensure_default_problem_persisted_reuses_existing_save():
    first = ensure_default_problem_persisted()
    second = ensure_default_problem_persisted()

    # default_problem_instance() stamps a fresh created_at on every call; if
    # ensure_default_problem_persisted() regenerated instead of loading the
    # existing save, this would differ between calls.
    assert second.created_at == first.created_at
