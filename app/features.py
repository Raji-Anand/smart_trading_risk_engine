import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.sort_values("date").copy()
    data["return_1d"] = data["close"].pct_change()
    data["return_5d"] = data["close"].pct_change(5)
    data["ma_10_ratio"] = data["close"] / data["close"].rolling(10).mean() - 1
    data["ma_30_ratio"] = data["close"] / data["close"].rolling(30).mean() - 1
    data["volatility_10d"] = data["return_1d"].rolling(10).std()
    data["volume_z"] = (data["volume"] - data["volume"].rolling(20).mean()) / data["volume"].rolling(20).std()
    data["target_up"] = (data["return_1d"].shift(-1) > 0).astype(int)
    data["target_return"] = data["return_1d"].shift(-1)
    return data.replace([np.inf, -np.inf], np.nan).dropna()


FEATURE_COLUMNS = ["return_1d", "return_5d", "ma_10_ratio", "ma_30_ratio", "volatility_10d", "volume_z"]
