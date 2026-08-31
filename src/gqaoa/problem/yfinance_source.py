"""`ProblemInstance` generator backed by real market data via yfinance.

The network call is isolated behind `_download_prices` so tests can mock it
without the `yfinance` package having to be installed (see the optional
`yfinance` extra in pyproject.toml). `yfinance` itself is imported lazily,
inside `_download_prices`, for the same reason.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gqaoa.problem.identifiers import compute_problem_id
from gqaoa.problem.instance import SCHEMA_VERSION, ProblemInstance
from gqaoa.problem.topology import Topology, generate_topology

ANNUALIZATION_FACTOR = 252


class YFinanceDataError(RuntimeError):
    """Raised when the requested tickers/date range don't yield usable data."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _download_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Adjusted close prices for `tickers` in [start_date, end_date), one
    column per ticker. Isolated for easy mocking in tests.
    """
    import yfinance as yf

    raw = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"]
    return raw[["Close"]].rename(columns={"Close": tickers[0]})


def generate_yfinance_problem(
    tickers: list[str],
    start_date: str,
    end_date: str,
    topology: Topology = "ring",
) -> ProblemInstance:
    prices = _download_prices(tickers, start_date, end_date)

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise YFinanceDataError(
            f"No price data returned for ticker(s) {missing} in range "
            f"[{start_date}, {end_date})"
        )

    prices = prices[tickers]
    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")

    if log_returns.empty or log_returns.isna().any().any():
        raise YFinanceDataError(
            f"Incomplete price data for ticker(s) {tickers} in range "
            f"[{start_date}, {end_date})"
        )

    n_assets = len(tickers)
    expected_value = (log_returns.mean() * ANNUALIZATION_FACTOR).tolist()
    cov_matrix = pd.DataFrame((log_returns.cov() * ANNUALIZATION_FACTOR).to_numpy())

    edges_hc = generate_topology(n_assets, topology)
    edges_hb = generate_topology(n_assets, topology)

    hash_params = {
        "tickers": sorted(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "topology": topology,
    }
    problem_id = compute_problem_id("yfinance", n_assets, hash_params)

    provenance = {
        "tickers": list(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "topology": topology,
        "annualization_factor": ANNUALIZATION_FACTOR,
    }

    return ProblemInstance(
        problem_id=problem_id,
        n_assets=n_assets,
        asset_names=list(tickers),
        expected_value=expected_value,
        cov_matrix=cov_matrix,
        edges_hc=edges_hc,
        edges_hb=edges_hb,
        source="yfinance",
        provenance=provenance,
        created_at=_now_iso(),
        schema_version=SCHEMA_VERSION,
    )
