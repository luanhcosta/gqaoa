import pytest

from gqaoa.domain.qaoa import QAOA


@pytest.fixture
def qaoa(toy_expected_value, toy_cov_matrix, toy_ring_edges):
    return QAOA(
        toy_expected_value, toy_cov_matrix,
        q=0.5, B=1, lamb=0,
        initial_state="dicke_state", mixture_layer="xy",
        edges_hc=toy_ring_edges, edges_hb=toy_ring_edges,
    )


def test_n_assets_and_edges(qaoa, toy_ring_edges):
    assert qaoa.n_assets == 4
    assert qaoa.edges_hc == toy_ring_edges
    assert qaoa.edges_hb == toy_ring_edges


def test_linear_weight_coefficients(qaoa):
    # cost_hamiltonian_wheight(i) = expected_value[i] + lamb*(2B - n_assets) - q*cov_matrix[i].sum()
    assert qaoa.cost_hamiltonian_wheight(0) == pytest.approx(0.08 - 0.5 * 1.35)
    assert qaoa.cost_hamiltonian_wheight(1) == pytest.approx(0.05 - 0.5 * 1.40)
    assert qaoa.cost_hamiltonian_wheight(2) == pytest.approx(0.10 - 0.5 * 1.30)
    assert qaoa.cost_hamiltonian_wheight(3) == pytest.approx(0.03 - 0.5 * 1.15)


def test_quadratic_weight_coefficients(qaoa):
    # cost_hamiltonian_wheight(i, j) = q*cov_matrix[i][j] + lamb
    assert qaoa.cost_hamiltonian_wheight(0, 1) == pytest.approx(0.5 * 0.20)
    assert qaoa.cost_hamiltonian_wheight(1, 2) == pytest.approx(0.5 * 0.15)
    assert qaoa.cost_hamiltonian_wheight(2, 3) == pytest.approx(0.5 * 0.05)
    assert qaoa.cost_hamiltonian_wheight(3, 0) == pytest.approx(0.5 * 0.05)


def test_qpu_call_counter_increments(qaoa, device_name):
    import numpy as np
    import pennylane as qml

    device = qml.device(device_name, wires=qaoa.n_assets)
    qnode = qml.QNode(qaoa.cost_function, device)

    assert qaoa.count_qpu_call == 0
    qnode(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    assert qaoa.count_qpu_call == 1
