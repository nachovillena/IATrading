import pandas as pd
import joblib
from pathlib import Path
from utils import extract_features

MODEL_DIR = Path("modelos")
SIGNALS_DIR = Path("senales")
SIGNALS_DIR.mkdir(exist_ok=True)

def generate_signals(periodo: str) -> None:
    raw = pd.DataFrame()  # implement load_data with periodo
    for model_file in MODEL_DIR.glob("modelo_*.pkl"):
        strategy = model_file.stem.split("_")[1]
        scaler = joblib.load(MODEL_DIR.parent / "scaler" / model_file.name.replace("modelo", "scaler"))
        model = joblib.load(model_file)
        X = extract_features(raw, strategy)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        df_signals = pd.DataFrame({
            "timestamp": raw.index,
            "signal": preds
        })
        df_signals.to_csv(SIGNALS_DIR / f"signals_{strategy}.csv", index=False)
