"""Synthetic `ProblemInstance` generator.

Deterministic given `seed`: builds a random PSD target covariance, simulates
a correlated daily log-return random walk from it, and derives
`expected_value`/`cov_matrix` from the empirical mean/covariance of that
simulated series (the same estimator convention used by the yfinance
generator), annualized by `annualization_factor=252`.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gqaoa.problem.identifiers import compute_problem_id
from gqaoa.problem.instance import SCHEMA_VERSION, ProblemInstance
from gqaoa.problem.topology import Topology, generate_topology

ANNUALIZATION_FACTOR = 252
_DEFAULT_TRADING_DAYS = 756  # ~3 trading years
_DAILY_VOL_SCALE = 0.02  # ~2% average daily volatility for the target covariance


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _random_psd_covariance(rng: np.random.Generator, n_assets: int, daily_vol_scale: float) -> np.ndarray:
    """A @ A.T is PSD by construction; rescaled so the average implied daily
    variance matches daily_vol_scale**2.
    """
    A = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T
    avg_var = np.trace(cov) / n_assets
    scale = (daily_vol_scale**2) / avg_var
    return cov * scale


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2


def generate_synthetic_problem(
    n_assets: int,
    seed: int | None = None,
    n_trading_days: int = _DEFAULT_TRADING_DAYS,
    topology: Topology = "ring",
) -> ProblemInstance:
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))

    rng = np.random.default_rng(seed)

    target_cov = _random_psd_covariance(rng, n_assets, _DAILY_VOL_SCALE)
    daily_mu = rng.uniform(0.0002, 0.0012, size=n_assets)
    daily_log_returns = rng.multivariate_normal(mean=daily_mu, cov=target_cov, size=n_trading_days)

    expected_value = (daily_log_returns.mean(axis=0) * ANNUALIZATION_FACTOR).tolist()
    cov_values = _symmetrize(np.cov(daily_log_returns, rowvar=False)) * ANNUALIZATION_FACTOR
    cov_matrix = pd.DataFrame(cov_values)

    asset_names = [f"asset_{i}" for i in range(n_assets)]
    edges_hc = generate_topology(n_assets, topology)
    edges_hb = generate_topology(n_assets, topology)

    hash_params = {
        "n_assets": n_assets,
        "seed": seed,
        "n_trading_days": n_trading_days,
        "topology": topology,
    }
    problem_id = compute_problem_id("synthetic", n_assets, hash_params)

    provenance = {
        "n_assets": n_assets,
        "seed": seed,
        "n_trading_days": n_trading_days,
        "topology": topology,
        "annualization_factor": ANNUALIZATION_FACTOR,
    }

    return ProblemInstance(
        problem_id=problem_id,
        n_assets=n_assets,
        asset_names=asset_names,
        expected_value=expected_value,
        cov_matrix=cov_matrix,
        edges_hc=edges_hc,
        edges_hb=edges_hb,
        source="synthetic",
        provenance=provenance,
        created_at=_now_iso(),
        schema_version=SCHEMA_VERSION,
    )
