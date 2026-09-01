"""Wraps the fixed 10-asset problem (`f_return_cov` + `RING_TOPOLOGY_EDGES`) as a
`ProblemInstance`, and is how `strategies/common.py::build_problem()` serves it
by default via `gqaoa.problem.store` instead of calling `f_return_cov()`
directly — same numbers, routed through the same persisted-problem mechanism as
every other problem source. `f_return_cov()`'s 10-asset order corresponds to
these tickers (not stored in `domain/data.py` itself, which predates
`gqaoa.problem`).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from gqaoa.config import DEFAULT_PROBLEM_ID, RING_TOPOLOGY_EDGES
from gqaoa.domain.data import f_return_cov
from gqaoa.problem.instance import SCHEMA_VERSION, ProblemInstance
from gqaoa.problem.store import load_problem, save_problem

ASSET_NAMES = ["AAPL", "MSFT", "NVDA", "AMD", "JNJ", "LLY", "UNH", "JPM", "BAC", "GS"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_problem_instance() -> ProblemInstance:
    expected_value, cov_matrix = f_return_cov()
    edges = [tuple(e) for e in RING_TOPOLOGY_EDGES]

    return ProblemInstance(
        problem_id=DEFAULT_PROBLEM_ID,
        n_assets=len(expected_value),
        asset_names=list(ASSET_NAMES),
        expected_value=list(expected_value),
        cov_matrix=pd.DataFrame(cov_matrix),
        edges_hc=edges,
        edges_hb=edges,
        source="default_fixed",
        provenance={"origin": "f_return_cov", "topology": "RING_TOPOLOGY_EDGES"},
        created_at=_now_iso(),
        schema_version=SCHEMA_VERSION,
    )


def ensure_default_problem_persisted() -> ProblemInstance:
    """Load the persisted default problem, generating+saving it on first use.

    `default_problem_instance()` stamps a fresh `created_at` on every call, so
    this checks for an existing save first rather than unconditionally
    generating and calling `save_problem()` (which would see a "different"
    payload — different `created_at` — on every call and either raise or
    rewrite the file). A fresh clone (or one where artifacts/problems/ was
    deleted) bootstraps the file on first call; every call after that just
    loads it, so the persisted `created_at` never changes.
    """
    try:
        return load_problem(DEFAULT_PROBLEM_ID)
    except FileNotFoundError:
        instance = default_problem_instance()
        save_problem(instance)
        return instance
