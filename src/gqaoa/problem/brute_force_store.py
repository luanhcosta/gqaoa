"""JSON persistence for brute-force results under
`artifacts/problems/<problem_id>/brute_force.json`.

Kept apart from `gqaoa.problem.store` (which owns `problem.json`) so that
module is left untouched: this file only adds a new artifact, `save_problem`/
`load_problem` are unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gqaoa.domain.brute_force import BruteForceResult
from gqaoa.paths import PROBLEMS_DIR

SCHEMA_VERSION = 1

_KEY_FIELDS = ("problem_id", "q", "B", "lamb", "search_mode")
# Fields excluded from the "is this the same result" comparison: `created_at`
# and `runtime_seconds` legitimately differ between two runs of an otherwise
# identical search, so they must not turn a would-be no-op into a conflict.
_VOLATILE_FIELDS = ("created_at", "runtime_seconds")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _brute_force_path(problem_id: str) -> Path:
    return PROBLEMS_DIR / problem_id / "brute_force.json"


def _serialize(problem_id: str, q: float, B: float, lamb: float, result: BruteForceResult, created_at: str) -> dict:
    return {
        "problem_id": problem_id,
        "search_mode": result.search_mode,
        "q": float(q),
        "B": float(B),
        "lamb": float(lamb),
        "optimal_bitstring": result.optimal_bitstring,
        "optimal_energy": float(result.optimal_energy),
        "n_candidates_evaluated": int(result.n_candidates_evaluated),
        "runtime_seconds": float(result.runtime_seconds),
        "created_at": created_at,
        "schema_version": SCHEMA_VERSION,
    }


def save_brute_force_result(
    problem_id: str,
    q: float,
    B: float,
    lamb: float,
    result: BruteForceResult,
    overwrite: bool = False,
) -> Path:
    """Write `result` to `artifacts/problems/<problem_id>/brute_force.json`.

    Keyed by `(problem_id, q, B, lamb, search_mode)`: if a file already exists
    there with the same key fields *and* the same result payload, this is a
    no-op. If it exists with different key fields or a different result,
    saving raises unless `overwrite=True`.
    """
    payload = _serialize(problem_id, q, B, lamb, result, created_at=_now_iso())
    path = _brute_force_path(problem_id)

    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None

        if existing is not None:
            comparable_existing = {k: v for k, v in existing.items() if k not in _VOLATILE_FIELDS}
            comparable_payload = {k: v for k, v in payload.items() if k not in _VOLATILE_FIELDS}
            if comparable_existing == comparable_payload:
                return path

            if not overwrite:
                differing_keys = [f for f in _KEY_FIELDS if existing.get(f) != payload[f]]
                if differing_keys:
                    raise FileExistsError(
                        f"Brute-force result for problem_id={problem_id!r} already exists at "
                        f"{path} with different {differing_keys}. Pass overwrite=True to replace it."
                    )
                raise FileExistsError(
                    f"Brute-force result for problem_id={problem_id!r} already exists at {path} "
                    "with different content. Pass overwrite=True to replace it."
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_brute_force_result(problem_id: str) -> dict:
    """Read back the brute-force result payload saved by `save_brute_force_result`."""
    path = _brute_force_path(problem_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved brute-force result found for problem_id={problem_id!r} at {path}"
        )
    return json.loads(path.read_text())
