from gqaoa.config import RING_TOPOLOGY_EDGES
from gqaoa.domain.data import f_return_cov
from gqaoa.problem.legacy import LEGACY_PROBLEM_ID, legacy_problem_instance


def test_legacy_problem_instance_matches_f_return_cov_and_ring_topology():
    expected_value, cov_matrix = f_return_cov()
    instance = legacy_problem_instance()

    assert instance.problem_id == LEGACY_PROBLEM_ID == "legacy-n10-fixed"
    assert instance.source == "legacy_fixed"
    assert instance.n_assets == 10
    assert instance.asset_names == [f"asset_{i}" for i in range(10)]
    assert instance.expected_value == list(expected_value)
    assert instance.cov_matrix.to_numpy().tolist() == cov_matrix.to_numpy().tolist()
    assert instance.edges_hc == [tuple(e) for e in RING_TOPOLOGY_EDGES]
    assert instance.edges_hb == [tuple(e) for e in RING_TOPOLOGY_EDGES]
    assert instance.schema_version == 1
