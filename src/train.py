import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from utils import load_features_labels

CONFIG_MASTER = Path("config_master.json")
SCALER_DIR = Path("scaler")
MODEL_DIR = Path("modelos")
SCALER_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def train_models() -> None:
    with open(CONFIG_MASTER) as f:
        master = json.load(f)
    for strategy, params_list in master.items():
        for idx, params in enumerate(params_list):
            X, y = load_features_labels(strategy, params)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
            pipe.fit(X, y)
            joblib.dump(pipe.named_steps["scaler"], SCALER_DIR / f"scaler_{strategy}_{idx}.pkl")
            joblib.dump(pipe.named_steps["clf"], MODEL_DIR / f"modelo_{strategy}_{idx}.pkl")
