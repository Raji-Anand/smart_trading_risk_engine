from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import MarketBar


def _synthetic_prices(symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
    end = end or datetime.utcnow().date()
    dates = pd.bdate_range(start=start, end=end)
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    drift = 0.0004 if "BTC" not in symbol else 0.0008
    vol = 0.018 if "BTC" not in symbol else 0.035
    returns = rng.normal(drift, vol, len(dates))
    close = 100 * np.exp(np.cumsum(returns))
    spread = np.maximum(close * rng.normal(0.006, 0.002, len(dates)), 0.01)

    return pd.DataFrame(
        {
            "date": dates.date,
            "symbol": symbol,
            "open": close * (1 + rng.normal(0, 0.003, len(dates))),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, len(dates)),
        }
    )


def fetch_market_data(symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(symbol, start=start.isoformat(), end=end, progress=False, auto_adjust=False)
        if raw.empty:
            return _synthetic_prices(symbol, start, end)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        return pd.DataFrame(
            {
                "date": pd.to_datetime(raw["Date"]).dt.date,
                "symbol": symbol,
                "open": raw["Open"].astype(float),
                "high": raw["High"].astype(float),
                "low": raw["Low"].astype(float),
                "close": raw["Close"].astype(float),
                "volume": raw["Volume"].fillna(0).astype(float),
            }
        ).dropna()
    except Exception:
        return _synthetic_prices(symbol, start, end)


def upsert_market_data(db: Session, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    symbol = str(frame["symbol"].iloc[0])
    dates = pd.to_datetime(frame["date"]).dt.date
    db.execute(delete(MarketBar).where(MarketBar.symbol == symbol, MarketBar.date.in_(dates.tolist())))
    rows = [
        MarketBar(
            symbol=row.symbol,
            date=row.date,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for row in frame.itertuples(index=False)
    ]
    db.add_all(rows)
    db.commit()
    return len(rows)


def load_prices(db: Session, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    stmt = select(MarketBar).where(MarketBar.symbol == symbol).order_by(MarketBar.date)
    if start:
        stmt = stmt.where(MarketBar.date >= start)
    if end:
        stmt = stmt.where(MarketBar.date <= end)
    rows = db.scalars(stmt).all()
    return pd.DataFrame(
        [
            {
                "date": row.date,
                "symbol": row.symbol,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
            for row in rows
        ]
    )


def ensure_symbol_data(db: Session, symbol: str, start: date = date(2021, 1, 1)) -> pd.DataFrame:
    prices = load_prices(db, symbol)
    if not prices.empty:
        return prices
    frame = fetch_market_data(symbol, start)
    upsert_market_data(db, frame)
    return load_prices(db, symbol)
