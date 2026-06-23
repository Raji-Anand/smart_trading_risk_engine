from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "market.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "BTC-USD"]
