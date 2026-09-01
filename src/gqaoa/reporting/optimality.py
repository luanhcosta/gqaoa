"""Compare QAOA run results against a persisted brute-force optimum.

Kept apart from `gqaoa.reporting.stats` (energy-only percentile stats) so that
module stays focused on statistics; this one is about solution-quality
comparisons against `gqaoa.problem.brute_force_store`.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from gqaoa.problem.brute_force_store import load_brute_force_result

# Energy gap (final_exp_val - optimal_energy) at or below this magnitude counts
# as "reached the optimum" — accounts for floating-point noise, not a
# meaningful suboptimality allowance.
ENERGY_MATCH_TOLERANCE = 1e-6


def derive_bitstring(probs) -> str:
    """Most-likely computational basis state, as a '0'/'1' string.

    `argmax(probs)` gives the state index under the same wire-order convention
    as `QAOA.probability_circuit`/`qml.probs` — wire 0 is the most significant
    bit (see `bitstring_energy`/`test_bitstring_energy_matches_real_qaoa_hamiltonian_matrix`),
    so the index is formatted with the standard `format(k, "0{n}b")`.
    """
    probs = np.asarray(probs)
    n_assets = int(round(math.log2(len(probs))))
    index = int(np.argmax(probs))
    return format(index, f"0{n_assets}b")


def try_load_brute_force(problem_id: str | None) -> dict | None:
    """Best-effort load of a persisted brute-force result.

    Returns None (never raises) when `problem_id` is None or when no
    brute-force result has been persisted for it — both are normal, expected
    states, not error conditions.
    """
    if problem_id is None:
        return None
    try:
        return load_brute_force_result(problem_id)
    except FileNotFoundError:
        return None


def compare_single_run_to_optimal(final_exp_val: float, probs, brute_force_result: dict) -> dict:
    """Compare one run's result against the persisted brute-force optimum."""
    found_bitstring = derive_bitstring(probs)
    optimal_bitstring = brute_force_result["optimal_bitstring"]
    optimal_energy = float(brute_force_result["optimal_energy"])
    energy_gap = float(final_exp_val) - optimal_energy

    return {
        "found_bitstring": found_bitstring,
        "optimal_bitstring": optimal_bitstring,
        "solution_matches_optimal": found_bitstring == optimal_bitstring,
        "energy_gap": energy_gap,
        "optimal_energy_reached": abs(energy_gap) <= ENERGY_MATCH_TOLERANCE,
    }


def compare_runs_to_optimal(
    bitstrings: Sequence[str], energies: Sequence[float], brute_force_result: dict
) -> dict:
    """Aggregate several runs' bitstrings/energies against the brute-force optimum.

    `solution_mode` is always a list: the most-common bitstring(s) found across
    runs (more than one entry on a tie), each occurring `solution_mode_count`
    times.
    """
    bitstrings = list(bitstrings)
    energies = list(energies)
    n_runs = len(bitstrings)

    counts = Counter(bitstrings)
    solution_mode_count = max(counts.values())
    solution_mode = sorted(bitstring for bitstring, count in counts.items() if count == solution_mode_count)

    optimal_bitstring = brute_force_result["optimal_bitstring"]
    optimal_energy = float(brute_force_result["optimal_energy"])
    energy_gaps = [float(e) - optimal_energy for e in energies]
    n_reached_optimum = sum(1 for gap in energy_gaps if abs(gap) <= ENERGY_MATCH_TOLERANCE)

    return {
        "solution_mode": solution_mode,
        "solution_mode_count": solution_mode_count,
        "n_runs": n_runs,
        "optimal_bitstring": optimal_bitstring,
        "solution_mode_matches_optimal": optimal_bitstring in solution_mode,
        "mean_energy_gap": float(np.mean(energy_gaps)),
        "optimal_energy_reached_rate": n_reached_optimum / n_runs,
    }
