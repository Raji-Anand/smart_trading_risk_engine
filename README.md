# Smart Trading & Risk Analytics Engine

A compact interview-ready fintech project: market data ingestion, SQL storage, quant strategies, ML price-direction signals, portfolio risk analytics, FastAPI endpoints, and a Streamlit dashboard.

## Architecture

- **Data layer:** pulls Yahoo Finance data when available, falls back to deterministic synthetic OHLCV data, and stores bars in SQLite through SQLAlchemy.
- **Strategy engine:** moving-average crossover, momentum, mean reversion, and simple arbitrage simulation.
- **ML layer:** feature engineering plus logistic-regression or random-forest classification for next-day up/down prediction.
- **Risk engine:** historical VaR, annualized volatility, max drawdown, exposure weights, and correlation matrix.
- **Backend API:** FastAPI endpoints for portfolio, backtest, prediction, and risk.
- **Dashboard:** Streamlit view for equity curves, metrics, predictions, and risk output.

## Quick Start

```bash
cd outputs/smart_trading_risk_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```bash
cd outputs/smart_trading_risk_engine
source .venv/bin/activate
streamlit run dashboard/streamlit_app.py
```

The API docs are available at `http://127.0.0.1:8000/docs`.

## Example API Calls

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT","BTC-USD"],"start":"2021-01-01"}'

curl -X POST http://127.0.0.1:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","strategy":"moving_average","initial_capital":100000}'

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL"}'

curl -X POST http://127.0.0.1:8000/risk \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT"],"weights":[0.6,0.4],"confidence":0.95}'
```

## Interview Talking Points

- Explain the separation between ingestion, persistence, strategy logic, ML features, risk, and API orchestration.
- Discuss how the synthetic data fallback makes demos deterministic while the same interface supports live data.
- Show why backtests shift signals by one day to avoid look-ahead bias.
- Mention production upgrades: PostgreSQL, Redis caching, async ingestion workers, model registry, CI, auth, portfolio accounting, and broker integration.

## Tests

```bash
pytest
```
