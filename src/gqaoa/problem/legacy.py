"""Wraps the existing fixed 10-asset problem (`f_return_cov` + `RING_TOPOLOGY_EDGES`)
as a `ProblemInstance`. Purely additive: does not change `data.py`, `config.py`,
or `build_problem()`'s existing usage of them.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from gqaoa.config import RING_TOPOLOGY_EDGES
from gqaoa.domain.data import f_return_cov
from gqaoa.problem.instance import SCHEMA_VERSION, ProblemInstance

LEGACY_PROBLEM_ID = "legacy-n10-fixed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def legacy_problem_instance() -> ProblemInstance:
    expected_value, cov_matrix = f_return_cov()
    n_assets = len(expected_value)
    asset_names = [f"asset_{i}" for i in range(n_assets)]
    edges = [tuple(e) for e in RING_TOPOLOGY_EDGES]

    return ProblemInstance(
        problem_id=LEGACY_PROBLEM_ID,
        n_assets=n_assets,
        asset_names=asset_names,
        expected_value=list(expected_value),
        cov_matrix=pd.DataFrame(cov_matrix),
        edges_hc=edges,
        edges_hb=edges,
        source="legacy_fixed",
        provenance={"origin": "f_return_cov", "topology": "RING_TOPOLOGY_EDGES"},
        created_at=_now_iso(),
        schema_version=SCHEMA_VERSION,
    )
