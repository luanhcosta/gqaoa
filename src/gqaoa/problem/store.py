"""JSON persistence for `ProblemInstance` under `artifacts/problems/<problem_id>/problem.json`."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gqaoa.paths import PROBLEMS_DIR
from gqaoa.problem.instance import ProblemInstance


def _serialize(instance: ProblemInstance) -> dict:
    cov_values = instance.cov_matrix
    cov_list = (
        cov_values.to_numpy().tolist()
        if isinstance(cov_values, pd.DataFrame)
        else [list(row) for row in cov_values]
    )
    return {
        "problem_id": instance.problem_id,
        "schema_version": instance.schema_version,
        "source": instance.source,
        "created_at": instance.created_at,
        "n_assets": instance.n_assets,
        "asset_names": list(instance.asset_names),
        "expected_value": [float(x) for x in instance.expected_value],
        "cov_matrix": cov_list,
        "edges_hc": [list(e) for e in instance.edges_hc],
        "edges_hb": [list(e) for e in instance.edges_hb],
        "provenance": instance.provenance,
    }


def _problem_path(problem_id: str) -> Path:
    return PROBLEMS_DIR / problem_id / "problem.json"


def save_problem(instance: ProblemInstance, overwrite: bool = False) -> Path:
    """Write `instance` to `artifacts/problems/<problem_id>/problem.json`.

    If a file already exists at that path: identical serialized content is a
    no-op (idempotent), different content raises unless `overwrite=True`.
    """
    payload = _serialize(instance)
    problem_path = _problem_path(instance.problem_id)

    if problem_path.exists():
        try:
            existing_payload = json.loads(problem_path.read_text())
        except json.JSONDecodeError:
            existing_payload = None

        if existing_payload == payload:
            return problem_path

        if not overwrite:
            raise FileExistsError(
                f"Problem '{instance.problem_id}' already exists at {problem_path} "
                "with different content. Pass overwrite=True to replace it."
            )

    problem_path.parent.mkdir(parents=True, exist_ok=True)
    problem_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return problem_path


def load_problem(problem_id: str) -> ProblemInstance:
    """Read back a `ProblemInstance` saved by `save_problem`.

    `cov_matrix` is reconstructed as a `pandas.DataFrame` with a plain
    RangeIndex, matching what `QAOA`/`compress_matrix` expect (see
    `ProblemInstance` docstring).
    """
    problem_path = _problem_path(problem_id)
    if not problem_path.exists():
        raise FileNotFoundError(
            f"No saved problem found for problem_id={problem_id!r} at {problem_path}"
        )

    payload = json.loads(problem_path.read_text())

    return ProblemInstance(
        problem_id=payload["problem_id"],
        n_assets=payload["n_assets"],
        asset_names=list(payload["asset_names"]),
        expected_value=[float(x) for x in payload["expected_value"]],
        cov_matrix=pd.DataFrame(payload["cov_matrix"]),
        edges_hc=[tuple(e) for e in payload["edges_hc"]],
        edges_hb=[tuple(e) for e in payload["edges_hb"]],
        source=payload["source"],
        provenance=payload["provenance"],
        created_at=payload["created_at"],
        schema_version=payload["schema_version"],
    )
