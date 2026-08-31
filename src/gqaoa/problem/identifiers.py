"""Deterministic `problem_id` derivation from generation parameters.

`problem_id` format: `<source>-n<N>-<hash8>`, where `<hash8>` is the first 8
hex chars of the SHA-256 digest of the canonical (sorted-key) JSON
representation of the *generation parameters* (never the resulting numeric
data) — see ProblemInstanceTicket spec for the exact per-source key sets.
"""
from __future__ import annotations

import hashlib
import json


def compute_problem_id(source: str, n_assets: int, hash_params: dict) -> str:
    canonical = json.dumps(hash_params, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"{source}-n{n_assets}-{digest}"
