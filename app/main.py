from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.config import DEFAULT_SYMBOLS
from app.data import ensure_symbol_data, fetch_market_data, load_prices, upsert_market_data
from app.database import get_session, init_db
from app.ml import train_predictor
from app.risk import portfolio_risk
from app.schemas import BacktestRequest, IngestRequest, PredictRequest, RiskRequest
from app.strategies import STRATEGIES, run_backtest

app = FastAPI(
    title="Smart Trading & Risk Analytics Engine",
    description="Mini Bloomberg-style market data, strategy, ML signal, and risk API.",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "default_symbols": DEFAULT_SYMBOLS}


@app.post("/ingest")
def ingest(request: IngestRequest, db: Session = Depends(get_session)) -> dict:
    loaded = {}
    for symbol in request.symbols:
        frame = fetch_market_data(symbol, request.start, request.end)
        loaded[symbol] = upsert_market_data(db, frame)
    return {"loaded_rows": loaded}


@app.get("/portfolio")
def portfolio(symbols: str = "AAPL,MSFT", db: Session = Depends(get_session)) -> dict:
    response = {}
    for symbol in [item.strip() for item in symbols.split(",") if item.strip()]:
        prices = ensure_symbol_data(db, symbol)
        latest = prices.sort_values("date").iloc[-1]
        response[symbol] = {
            "last_date": str(latest["date"]),
            "last_close": float(latest["close"]),
            "bars": int(len(prices)),
        }
    return response


@app.post("/backtest")
def backtest(request: BacktestRequest, db: Session = Depends(get_session)) -> dict:
    if request.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
    prices = load_prices(db, request.symbol, request.start, request.end)
    if prices.empty:
        prices = ensure_symbol_data(db, request.symbol)
    try:
        result = run_backtest(prices, request.strategy, request.initial_capital)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"symbol": request.symbol, "strategy": request.strategy, **result}


@app.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_session)) -> dict:
    prices = ensure_symbol_data(db, request.symbol)
    try:
        result = train_predictor(prices)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"symbol": request.symbol, "horizon_days": request.horizon_days, **result}


@app.post("/risk")
def risk(request: RiskRequest, db: Session = Depends(get_session)) -> dict:
    try:
        return portfolio_risk(db, request.symbols, request.weights, request.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
