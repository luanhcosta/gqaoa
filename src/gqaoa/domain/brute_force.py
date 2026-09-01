"""Classical (non-quantum) evaluation of the QAOA cost Hamiltonian.

Replicates `QAOA.cost_hamiltonian_wheight`/`QAOA.H_C` (see `gqaoa.domain.qaoa`)
without touching PennyLane, so it can run on plain CPU with no quantum
dependency. Kept out of `qaoa.py` on purpose: this module must stay usable by
the `gqaoa-brute-force` CLI without importing PennyLane at runtime.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

SearchMode = Literal["full", "fixed-cardinality"]


def _row_sum(cov_matrix, i: int) -> float:
    """`cov_matrix[i].sum()`, matching `QAOA.cost_hamiltonian_wheight`'s indexing:
    for a `pandas.DataFrame` with a plain RangeIndex, `cov_matrix[i]` selects
    *column* i; for an array-like, `cov_matrix[i]` selects *row* i. Both give
    the same result for the symmetric covariance matrices this module operates
    on, but each branch mirrors the original indexing on its own input type.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        return float(cov_matrix[i].sum())
    return float(np.asarray(cov_matrix)[i].sum())


def _entry(cov_matrix, i: int, j: int) -> float:
    """`cov_matrix[i][j]`, matching `QAOA.cost_hamiltonian_wheight`'s indexing
    (see `_row_sum` for why the DataFrame/array-like cases are handled apart).
    """
    if isinstance(cov_matrix, pd.DataFrame):
        return float(cov_matrix[i][j])
    return float(np.asarray(cov_matrix)[i][j])


def bitstring_energy(
    bitstring: str,
    expected_value: Sequence[float],
    cov_matrix,
    q: float,
    B: float,
    lamb: float,
    edges_hc: Sequence[tuple[int, int]],
) -> float:
    """Classical energy of `bitstring` under `QAOA.H_C`, computed without PennyLane.

    `bitstring` is a string of '0'/'1' characters (one per qubit/asset), mapped
    to Z-eigenvalues via `z_k = 1 - 2*bit_k` (bit 0 -> +1, bit 1 -> -1: the
    `PauliZ` eigenvalue convention on the computational basis `|0>`/`|1>`).

    The field (single-Z) term is summed over `range(4)` only, never
    `range(n_assets)` — this replicates `QAOA.__init__`'s `H_C1` construction
    verbatim (see `gqaoa.domain.qaoa.QAOA`); intentional, not a bug, and not
    generalized here. `bitstring`/`expected_value`/`cov_matrix` must therefore
    cover at least 4 assets.
    """
    n_assets = len(expected_value)
    z = [1 - 2 * int(bit) for bit in bitstring]

    energy = 0.0
    for i, j in edges_hc:
        weight = q * _entry(cov_matrix, i, j) + lamb
        energy += weight * z[i] * z[j]

    for i in range(4):
        weight = expected_value[i] + lamb * (2 * B - n_assets) - q * _row_sum(cov_matrix, i)
        energy += weight * z[i]

    return energy


@dataclass
class BruteForceResult:
    optimal_bitstring: str
    optimal_energy: float
    search_mode: SearchMode
    n_candidates_evaluated: int
    runtime_seconds: float


def _full_candidates(n_assets: int):
    for k in range(2**n_assets):
        yield format(k, f"0{n_assets}b")


def _fixed_cardinality_candidates(n_assets: int, B: int):
    for ones in itertools.combinations(range(n_assets), B):
        ones_set = set(ones)
        yield "".join("1" if i in ones_set else "0" for i in range(n_assets))


def brute_force_search(
    expected_value: Sequence[float],
    cov_matrix,
    q: float,
    B: float,
    lamb: float,
    edges_hc: Sequence[tuple[int, int]],
    n_assets: int,
    search_mode: SearchMode = "fixed-cardinality",
) -> BruteForceResult:
    """Exhaustively evaluate `bitstring_energy` and return the minimizer.

    `search_mode="full"` enumerates all `2**n_assets` bitstrings.
    `search_mode="fixed-cardinality"` restricts the search to bitstrings with
    exactly `B` ones (`itertools.combinations(range(n_assets), B)`).
    """
    if search_mode == "full":
        candidates = _full_candidates(n_assets)
    elif search_mode == "fixed-cardinality":
        candidates = _fixed_cardinality_candidates(n_assets, int(B))
    else:
        raise ValueError(
            f"Unknown search_mode: {search_mode!r} (expected 'full' or 'fixed-cardinality')"
        )

    start = time.perf_counter()
    best_bitstring: str | None = None
    best_energy: float | None = None
    n_evaluated = 0

    for bitstring in candidates:
        energy = bitstring_energy(bitstring, expected_value, cov_matrix, q, B, lamb, edges_hc)
        n_evaluated += 1
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    runtime_seconds = time.perf_counter() - start

    if best_bitstring is None or best_energy is None:
        raise ValueError(
            f"No candidates to evaluate for search_mode={search_mode!r}, "
            f"n_assets={n_assets}, B={B}"
        )

    return BruteForceResult(
        optimal_bitstring=best_bitstring,
        optimal_energy=best_energy,
        search_mode=search_mode,
        n_candidates_evaluated=n_evaluated,
        runtime_seconds=runtime_seconds,
    )
