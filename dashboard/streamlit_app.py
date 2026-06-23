import requests
import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Smart Trading & Risk Analytics", layout="wide")
API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
st.title("Smart Trading & Risk Analytics Engine")

symbol = st.sidebar.selectbox("Symbol", ["AAPL", "MSFT", "BTC-USD", "ETH-USD", "NVDA"])
strategy = st.sidebar.selectbox("Strategy", ["moving_average", "momentum", "mean_reversion", "arbitrage"])
capital = st.sidebar.number_input("Initial capital", min_value=10_000, value=100_000, step=10_000)

if st.sidebar.button("Ingest / refresh data"):
    response = requests.post(f"{API_URL}/ingest", json={"symbols": [symbol], "start": "2021-01-01"}, timeout=60)
    st.sidebar.success(response.json())

left, middle, right = st.columns(3)

portfolio = requests.get(f"{API_URL}/portfolio", params={"symbols": symbol}, timeout=30).json()
latest = portfolio[symbol]
left.metric("Latest close", f"${latest['last_close']:,.2f}")
middle.metric("Market bars", latest["bars"])
right.metric("Latest date", latest["last_date"])

backtest = requests.post(
    f"{API_URL}/backtest",
    json={"symbol": symbol, "strategy": strategy, "initial_capital": capital},
    timeout=60,
).json()

metrics = backtest["metrics"]
cols = st.columns(4)
cols[0].metric("PnL", f"${metrics['pnl']:,.0f}")
cols[1].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
cols[2].metric("Volatility", f"{metrics['volatility']:.1%}")
cols[3].metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")

curve = pd.DataFrame(backtest["equity_curve"])
fig = px.line(curve, x="date", y=["equity", "buy_hold_equity"], title="Strategy vs Buy-and-Hold Equity")
st.plotly_chart(fig, use_container_width=True)

prediction = requests.post(f"{API_URL}/predict", json={"symbol": symbol}, timeout=60).json()
risk = requests.post(f"{API_URL}/risk", json={"symbols": [symbol]}, timeout=60).json()

ml_col, risk_col = st.columns(2)
ml_col.subheader("ML Signal")
ml_col.json(prediction)
risk_col.subheader("Risk")
risk_col.json(risk)
