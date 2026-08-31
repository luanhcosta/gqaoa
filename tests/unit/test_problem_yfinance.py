import numpy as np
import pandas as pd
import pytest

from gqaoa.problem import yfinance_source
from gqaoa.problem.yfinance_source import YFinanceDataError, generate_yfinance_problem


def _fake_prices(tickers, n_days=60, seed=0, all_nan_for=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = {}
    for t in tickers:
        if all_nan_for and t in all_nan_for:
            data[t] = np.full(n_days, np.nan)
        else:
            log_returns = rng.normal(0.0005, 0.01, size=n_days)
            data[t] = 100 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(data, index=dates)


def test_generate_yfinance_problem_success(monkeypatch):
    tickers = ["AAA", "BBB", "CCC"]
    prices = _fake_prices(tickers)
    monkeypatch.setattr(yfinance_source, "_download_prices", lambda t, s, e: prices)

    instance = generate_yfinance_problem(tickers, "2024-01-01", "2024-03-31")

    assert instance.n_assets == 3
    assert instance.asset_names == tickers
    assert len(instance.expected_value) == 3
    assert instance.cov_matrix.shape == (3, 3)
    values = instance.cov_matrix.to_numpy()
    assert np.allclose(values, values.T)
    assert instance.source == "yfinance"
    assert instance.provenance["tickers"] == tickers
    assert instance.provenance["start_date"] == "2024-01-01"
    assert instance.provenance["end_date"] == "2024-03-31"
    assert instance.provenance["annualization_factor"] == 252


def test_generate_yfinance_problem_missing_ticker_raises_clear_error(monkeypatch):
    tickers = ["AAA", "ZZZ"]
    prices = _fake_prices(tickers, all_nan_for=["ZZZ"])
    monkeypatch.setattr(yfinance_source, "_download_prices", lambda t, s, e: prices)

    with pytest.raises(YFinanceDataError, match="ZZZ"):
        generate_yfinance_problem(tickers, "2024-01-01", "2024-03-31")


def test_problem_id_is_stable_regardless_of_ticker_order(monkeypatch):
    tickers_a = ["AAA", "BBB", "CCC"]
    tickers_b = ["CCC", "AAA", "BBB"]
    prices = _fake_prices(["AAA", "BBB", "CCC"])
    monkeypatch.setattr(yfinance_source, "_download_prices", lambda t, s, e: prices)

    a = generate_yfinance_problem(tickers_a, "2024-01-01", "2024-03-31")
    b = generate_yfinance_problem(tickers_b, "2024-01-01", "2024-03-31")

    assert a.problem_id == b.problem_id
    assert a.asset_names == tickers_a
    assert b.asset_names == tickers_b
