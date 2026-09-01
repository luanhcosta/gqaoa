import numpy as np
import pytest

from gqaoa.problem import brute_force_store
from gqaoa.reporting.optimality import (
    ENERGY_MATCH_TOLERANCE,
    compare_runs_to_optimal,
    compare_single_run_to_optimal,
    derive_bitstring,
    try_load_brute_force,
)


def _brute_force_result(bitstring="0110", energy=-1.5):
    return {
        "optimal_bitstring": bitstring,
        "optimal_energy": energy,
        "q": 0.5,
        "B": 2,
        "lamb": 0,
        "search_mode": "full",
    }


def test_derive_bitstring_picks_argmax_with_msb_first_convention():
    # 4-qubit state space (2**4 = 16 probs); index 6 -> "0110" under
    # wire-0-is-most-significant-bit convention (see
    # test_bitstring_energy_matches_real_qaoa_hamiltonian_matrix).
    probs = np.zeros(16)
    probs[6] = 0.9
    assert derive_bitstring(probs) == "0110"


def test_derive_bitstring_all_zeros_state():
    probs = np.zeros(8)
    probs[0] = 1.0
    assert derive_bitstring(probs) == "000"


def test_compare_single_run_to_optimal_when_solution_matches():
    probs = np.zeros(16)
    probs[6] = 1.0  # "0110"
    brute_force_result = _brute_force_result(bitstring="0110", energy=-1.5)

    comparison = compare_single_run_to_optimal(-1.5, probs, brute_force_result)

    assert comparison["found_bitstring"] == "0110"
    assert comparison["optimal_bitstring"] == "0110"
    assert comparison["solution_matches_optimal"] is True
    assert comparison["energy_gap"] == pytest.approx(0.0)
    assert comparison["optimal_energy_reached"] is True


def test_compare_single_run_to_optimal_when_solution_does_not_match():
    probs = np.zeros(16)
    probs[3] = 1.0  # "0011"
    brute_force_result = _brute_force_result(bitstring="0110", energy=-1.5)

    comparison = compare_single_run_to_optimal(-1.2, probs, brute_force_result)

    assert comparison["found_bitstring"] == "0011"
    assert comparison["solution_matches_optimal"] is False
    assert comparison["energy_gap"] == pytest.approx(0.3)
    assert comparison["optimal_energy_reached"] is False


def test_compare_single_run_to_optimal_respects_energy_match_tolerance():
    probs = np.zeros(16)
    probs[6] = 1.0
    brute_force_result = _brute_force_result(bitstring="1111", energy=-1.5)

    # Bitstring doesn't match, but energy is within tolerance of optimal.
    comparison = compare_single_run_to_optimal(-1.5 + ENERGY_MATCH_TOLERANCE / 2, probs, brute_force_result)

    assert comparison["solution_matches_optimal"] is False
    assert comparison["optimal_energy_reached"] is True


def test_compare_runs_to_optimal_single_mode_matches_optimal():
    bitstrings = ["0110", "0110", "0110", "0011"]
    energies = [-1.5, -1.5, -1.4, -1.0]
    brute_force_result = _brute_force_result(bitstring="0110", energy=-1.5)

    result = compare_runs_to_optimal(bitstrings, energies, brute_force_result)

    assert result["solution_mode"] == ["0110"]
    assert result["solution_mode_count"] == 3
    assert result["n_runs"] == 4
    assert result["optimal_bitstring"] == "0110"
    assert result["solution_mode_matches_optimal"] is True
    assert result["mean_energy_gap"] == pytest.approx(
        sum(e - (-1.5) for e in energies) / 4
    )
    assert result["optimal_energy_reached_rate"] == pytest.approx(2 / 4)


def test_compare_runs_to_optimal_returns_all_tied_bitstrings_as_a_list():
    bitstrings = ["0110", "0011", "0110", "0011"]
    energies = [-1.5, -1.5, -1.4, -1.4]
    brute_force_result = _brute_force_result(bitstring="1111", energy=-1.5)

    result = compare_runs_to_optimal(bitstrings, energies, brute_force_result)

    assert result["solution_mode"] == ["0011", "0110"]
    assert result["solution_mode_count"] == 2
    assert result["solution_mode_matches_optimal"] is False


@pytest.fixture(autouse=True)
def _isolated_problems_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(brute_force_store, "PROBLEMS_DIR", tmp_path / "problems")
    yield


def test_try_load_brute_force_returns_none_for_none_problem_id():
    assert try_load_brute_force(None) is None


def test_try_load_brute_force_returns_none_when_no_result_persisted():
    assert try_load_brute_force("no-such-problem") is None


def test_try_load_brute_force_returns_payload_when_persisted():
    from gqaoa.domain.brute_force import BruteForceResult
    from gqaoa.problem.brute_force_store import save_brute_force_result

    result = BruteForceResult(
        optimal_bitstring="0110", optimal_energy=-1.5, search_mode="full",
        n_candidates_evaluated=16, runtime_seconds=0.01,
    )
    save_brute_force_result("problem-xyz", q=0.5, B=2, lamb=0.0, result=result)

    payload = try_load_brute_force("problem-xyz")

    assert payload is not None
    assert payload["optimal_bitstring"] == "0110"
    assert payload["optimal_energy"] == pytest.approx(-1.5)
