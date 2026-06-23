import numpy as np
import pandas as pd

from app.metrics import max_drawdown, sharpe_ratio
from app.strategies import run_backtest


def sample_prices(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=rows)
    close = 100 + np.cumsum(np.sin(np.arange(rows) / 8) + 0.2)
    return pd.DataFrame(
        {
            "date": dates.date,
            "symbol": "TEST",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_sharpe_ratio_positive_for_positive_returns():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_detects_drop():
    equity = pd.Series([100, 120, 90, 110])
    assert round(max_drawdown(equity), 2) == -0.25


def test_backtest_returns_metrics_and_curve():
    result = run_backtest(sample_prices(), "moving_average", 100_000)
    assert result["metrics"]["final_equity"] > 0
    assert len(result["equity_curve"]) == 180
