import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.data import ensure_symbol_data
from app.metrics import annualized_volatility, historical_var, max_drawdown


def portfolio_risk(db: Session, symbols: list[str], weights: list[float] | None = None, confidence: float = 0.95) -> dict:
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if weights is None:
        weights = [1 / len(symbols)] * len(symbols)
    if len(weights) != len(symbols):
        raise ValueError("weights must have the same length as symbols.")

    normalized = np.array(weights, dtype=float)
    normalized = normalized / normalized.sum()

    returns = []
    for symbol in symbols:
        prices = ensure_symbol_data(db, symbol)
        series = prices.sort_values("date").set_index("date")["close"].pct_change().rename(symbol)
        returns.append(series)

    matrix = pd.concat(returns, axis=1).dropna()
    portfolio_returns = matrix.dot(normalized)
    equity = (1 + portfolio_returns).cumprod()
    exposures = {symbol: float(weight) for symbol, weight in zip(symbols, normalized)}

    return {
        "symbols": symbols,
        "exposures": exposures,
        "confidence": confidence,
        "value_at_risk": historical_var(portfolio_returns, confidence),
        "portfolio_volatility": annualized_volatility(portfolio_returns),
        "max_drawdown": max_drawdown(equity),
        "correlation": matrix.corr().round(3).to_dict(),
    }
