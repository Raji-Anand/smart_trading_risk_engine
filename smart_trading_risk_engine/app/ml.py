from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_COLUMNS, build_features


def train_predictor(prices, model_type: str = "random_forest") -> dict:
    data = build_features(prices)
    if len(data) < 100:
        raise ValueError("At least 100 bars are required to train the ML model.")

    x = data[FEATURE_COLUMNS]
    y = data["target_up"]
    split = int(len(data) * 0.75)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == "logistic":
        model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    else:
        model = RandomForestClassifier(n_estimators=250, min_samples_leaf=5, random_state=42)

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    latest_features = x.iloc[[-1]]
    latest_probability = float(model.predict_proba(latest_features)[:, 1][0])

    try:
        auc = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        auc = 0.0

    return {
        "model_type": model_type,
        "observations": int(len(data)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_auc": auc,
        "latest_date": str(data["date"].iloc[-1]),
        "probability_up": latest_probability,
        "signal": "bullish" if latest_probability >= 0.55 else "bearish" if latest_probability <= 0.45 else "neutral",
    }
