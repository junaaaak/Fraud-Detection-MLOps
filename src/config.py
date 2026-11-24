from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "database.db"
TRAINING_CSV = DATA_DIR / "training_data.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
