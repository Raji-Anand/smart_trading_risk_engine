import numpy as np
import pandas as pd


TRADING_DAYS = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def annualized_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return float(drawdown.min())


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return float(-np.quantile(returns.dropna(), 1 - confidence))


def performance_summary(equity: pd.Series) -> dict[str, float]:
    returns = returns_from_equity(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio(returns),
        "volatility": annualized_volatility(returns),
        "max_drawdown": max_drawdown(equity),
    }
