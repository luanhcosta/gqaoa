from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SCHEMA_VERSION = 1


@dataclass
class ProblemInstance:
    """A portfolio-optimization problem instance ready to feed into QAOA.

    `cov_matrix` is kept as a `pandas.DataFrame` with a plain RangeIndex
    (0..n_assets-1) for both axes, matching the shape `f_return_cov()` has
    always produced — `QAOA.cost_hamiltonian_wheight` and
    `compression.compress_matrix` both index it positionally
    (`cov_matrix[i][j]`, `cov_matrix[i].sum()`).
    """

    problem_id: str
    n_assets: int
    asset_names: list[str]
    expected_value: list[float]
    cov_matrix: pd.DataFrame
    edges_hc: list[tuple[int, int]]
    edges_hb: list[tuple[int, int]]
    source: Literal["synthetic", "yfinance", "legacy_fixed"]
    provenance: dict
    created_at: str
    schema_version: int = SCHEMA_VERSION
