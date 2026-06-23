import numpy as np
import pandas as pd

from app.metrics import performance_summary


def _prepare(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.sort_values("date").copy()
    data["return"] = data["close"].pct_change().fillna(0)
    return data


def moving_average_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    data = _prepare(prices)
    fast_ma = data["close"].rolling(fast).mean()
    slow_ma = data["close"].rolling(slow).mean()
    return pd.Series(np.where(fast_ma > slow_ma, 1, 0), index=data.index).shift(1).fillna(0)


def momentum(prices: pd.DataFrame, lookback: int = 20) -> pd.Series:
    data = _prepare(prices)
    signal = np.where(data["close"].pct_change(lookback) > 0, 1, 0)
    return pd.Series(signal, index=data.index).shift(1).fillna(0)


def mean_reversion(prices: pd.DataFrame, window: int = 20, z_entry: float = -1.0) -> pd.Series:
    data = _prepare(prices)
    mean = data["close"].rolling(window).mean()
    std = data["close"].rolling(window).std()
    z_score = (data["close"] - mean) / std
    signal = np.where(z_score < z_entry, 1, np.where(z_score > abs(z_entry), 0, np.nan))
    return pd.Series(signal, index=data.index).ffill().shift(1).fillna(0)


def arbitrage_simulation(prices: pd.DataFrame) -> pd.Series:
    data = _prepare(prices)
    intraday_gap = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)
    signal = np.where(intraday_gap < -0.01, 1, 0)
    return pd.Series(signal, index=data.index).shift(1).fillna(0)


STRATEGIES = {
    "moving_average": moving_average_crossover,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "arbitrage": arbitrage_simulation,
}


def run_backtest(prices: pd.DataFrame, strategy: str, initial_capital: float = 100_000) -> dict:
    if prices.empty or len(prices) < 60:
        raise ValueError("At least 60 price bars are required for a meaningful backtest.")
    data = _prepare(prices)
    positions = STRATEGIES[strategy](data)
    data["signal"] = positions
    data["strategy_return"] = data["signal"] * data["return"]
    data["equity"] = initial_capital * (1 + data["strategy_return"]).cumprod()
    data["buy_hold_equity"] = initial_capital * (1 + data["return"]).cumprod()

    curve = data[["date", "close", "signal", "strategy_return", "equity", "buy_hold_equity"]].copy()
    curve["date"] = curve["date"].astype(str)
    summary = performance_summary(curve["equity"])
    summary["final_equity"] = float(curve["equity"].iloc[-1])
    summary["pnl"] = float(curve["equity"].iloc[-1] - initial_capital)
    return {"metrics": summary, "equity_curve": curve.to_dict(orient="records")}
