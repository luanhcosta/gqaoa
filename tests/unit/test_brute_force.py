import numpy as np
import pennylane as qml
import pytest

from gqaoa.domain.brute_force import BruteForceResult, bitstring_energy, brute_force_search


def test_bitstring_energy_matches_hand_calculated_weights(toy_expected_value, toy_cov_matrix, toy_ring_edges):
    # q=0.5, B=1, lamb=0, toy 4-asset ring problem (see tests/conftest.py).
    # linear weight(i) = expected_value[i] - 0.5*cov_matrix[i].sum():
    #   weight(0) = 0.08 - 0.5*1.35 = -0.595
    #   weight(1) = 0.05 - 0.5*1.40 = -0.65
    #   weight(2) = 0.10 - 0.5*1.30 = -0.55
    #   weight(3) = 0.03 - 0.5*1.15 = -0.545
    # quadratic weight(i, j) = 0.5*cov_matrix[i][j]:
    #   weight(0,1) = 0.10, weight(1,2) = 0.075, weight(2,3) = 0.025, weight(3,0) = 0.025
    # bitstring "0110" -> z = [+1, -1, -1, +1]
    #   linear:    -0.595*1 + -0.65*-1 + -0.55*-1 + -0.545*1 = 0.06
    #   quadratic: 0.10*(1*-1) + 0.075*(-1*-1) + 0.025*(-1*1) + 0.025*(1*1) = -0.025
    #   total: 0.06 - 0.025 = 0.035
    energy = bitstring_energy(
        "0110", toy_expected_value, toy_cov_matrix,
        q=0.5, B=1, lamb=0, edges_hc=toy_ring_edges,
    )
    assert energy == pytest.approx(0.035)


def test_bitstring_energy_all_zeros_and_all_ones_split_into_linear_and_quadratic_parts(
    toy_expected_value, toy_cov_matrix, toy_ring_edges
):
    kwargs = dict(
        expected_value=toy_expected_value, cov_matrix=toy_cov_matrix,
        q=0.5, B=1, lamb=0, edges_hc=toy_ring_edges,
    )
    energy_all_zeros = bitstring_energy("0000", **kwargs)  # all z = +1
    energy_all_ones = bitstring_energy("1111", **kwargs)  # all z = -1

    # ZZ terms are invariant under a global sign flip of z; Z terms flip sign.
    # So all_zeros = quadratic + linear, all_ones = quadratic - linear.
    linear_part = sum(toy_expected_value[i] - 0.5 * toy_cov_matrix[i].sum() for i in range(4))
    quadratic_part = 0.5 * (0.20 + 0.15 + 0.05 + 0.05)  # weight(i,j) summed over the ring edges

    assert (energy_all_zeros - energy_all_ones) == pytest.approx(2 * linear_part)
    assert (energy_all_zeros + energy_all_ones) == pytest.approx(2 * quadratic_part)


def test_brute_force_search_full_mode_finds_expected_minimum(toy_expected_value, toy_cov_matrix, toy_ring_edges):
    q, B, lamb = 0.5, 1, 0
    energies = {
        format(k, "04b"): bitstring_energy(
            format(k, "04b"), toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb, edges_hc=toy_ring_edges,
        )
        for k in range(16)
    }
    expected_bitstring = min(energies, key=energies.get)
    expected_energy = energies[expected_bitstring]

    result = brute_force_search(
        toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb,
        edges_hc=toy_ring_edges, n_assets=4, search_mode="full",
    )

    assert isinstance(result, BruteForceResult)
    assert result.search_mode == "full"
    assert result.n_candidates_evaluated == 16
    assert result.optimal_bitstring == expected_bitstring
    assert result.optimal_energy == pytest.approx(expected_energy)
    assert result.runtime_seconds >= 0


def test_brute_force_search_fixed_cardinality_only_evaluates_matching_bitstrings(
    toy_expected_value, toy_cov_matrix, toy_ring_edges
):
    q, B, lamb = 0.5, 2, 0
    result = brute_force_search(
        toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb,
        edges_hc=toy_ring_edges, n_assets=4, search_mode="fixed-cardinality",
    )

    from math import comb
    assert result.n_candidates_evaluated == comb(4, B)
    assert result.optimal_bitstring.count("1") == B


def test_fixed_cardinality_minimum_is_not_better_than_full_minimum(toy_expected_value, toy_cov_matrix, toy_ring_edges):
    q, B, lamb = 0.5, 2, 0

    full_result = brute_force_search(
        toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb,
        edges_hc=toy_ring_edges, n_assets=4, search_mode="full",
    )
    fixed_result = brute_force_search(
        toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb,
        edges_hc=toy_ring_edges, n_assets=4, search_mode="fixed-cardinality",
    )

    # full searches a superset of fixed-cardinality's candidates, so its minimum
    # can only be <= the minimum restricted to exactly-B-ones bitstrings.
    assert full_result.optimal_energy <= fixed_result.optimal_energy


def test_brute_force_search_rejects_unknown_search_mode(toy_expected_value, toy_cov_matrix, toy_ring_edges):
    with pytest.raises(ValueError, match="search_mode"):
        brute_force_search(
            toy_expected_value, toy_cov_matrix, q=0.5, B=1, lamb=0,
            edges_hc=toy_ring_edges, n_assets=4, search_mode="bogus",
        )


def test_bitstring_energy_matches_real_qaoa_hamiltonian_matrix(toy_expected_value, toy_cov_matrix, toy_ring_edges):
    from gqaoa.domain.qaoa import QAOA

    q, B, lamb = 0.5, 1, 0
    qaoa = QAOA(
        toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb,
        edges_hc=toy_ring_edges, edges_hb=toy_ring_edges,
    )
    n = qaoa.n_assets

    # H_C is diagonal in the computational basis (only Z / ZZ terms), so its
    # diagonal directly gives the energy of each computational basis state —
    # index k corresponds to bitstring format(k, "0{n}b") under PennyLane's
    # wire-order convention (wire 0 = most significant bit).
    matrix = qml.matrix(qaoa.H_C, wire_order=list(range(n)))
    diagonal = np.diag(matrix).real

    for k in range(2**n):
        bitstring = format(k, f"0{n}b")
        expected = diagonal[k]
        actual = bitstring_energy(
            bitstring, toy_expected_value, toy_cov_matrix, q=q, B=B, lamb=lamb, edges_hc=toy_ring_edges,
        )
        assert actual == pytest.approx(expected)
