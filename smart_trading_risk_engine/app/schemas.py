from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "BTC-USD"])
    start: date = date(2021, 1, 1)
    end: date | None = None


class BacktestRequest(BaseModel):
    symbol: str = "AAPL"
    strategy: Literal["moving_average", "momentum", "mean_reversion", "arbitrage"] = "moving_average"
    start: date | None = None
    end: date | None = None
    initial_capital: float = 100_000


class PredictRequest(BaseModel):
    symbol: str = "AAPL"
    horizon_days: int = 1


class RiskRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT"])
    weights: list[float] | None = None
    confidence: float = 0.95
