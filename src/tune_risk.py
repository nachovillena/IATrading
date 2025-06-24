import json
import numpy as np
from pathlib import Path
from utils import backtest_signals

SIGNALS_DIR = Path("senales")
TUNE_DIR = Path("tuning")
TUNE_DIR.mkdir(exist_ok=True)

def tune_risk_confidence() -> None:
    for sig_file in SIGNALS_DIR.glob("signals_*.csv"):
        strategy = sig_file.stem.split("_")[1]
        df = backtest_signals(None, strategy, {})
        best = {"sharpe": -np.inf}
        for risk in np.linspace(0.001, 0.02, 10):
            for conf in np.linspace(0.5, 0.9, 5):
                metrics = simulate_with_risk(df, risk, conf)
                if metrics["sharpe_ratio"] > best["sharpe"]:
                    best = {"risk": risk, "confidence": conf, "sharpe": metrics["sharpe_ratio"]}
        with open(TUNE_DIR / f"tune_{strategy}.json", "w") as f:
            json.dump(best, f, indent=2)

def simulate_with_risk(signals, risk, confidence):
    return {"sharpe_ratio": np.random.rand()}
