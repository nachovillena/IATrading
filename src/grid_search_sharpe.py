import json
import numpy as np
from pathlib import Path
from utils import backtest_signals

TUNE_FINE_DIR = Path("tuning_fino")
TUNE_FINE_DIR.mkdir(exist_ok=True)

def grid_search_sharpe() -> None:
    for _ in Path("parameters").glob("param_grid_*.csv"):
        strategy = _.stem.replace("param_grid_", "")
        best = {"sharpe": -np.inf}
        for sl in np.linspace(0.005, 0.02, 5):
            for tp in np.linspace(0.01, 0.05, 5):
                for h in [10, 20, 50]:
                    metrics = backtest_sharpe_metrics(strategy, sl, tp, h)
                    if metrics["sharpe_ratio"] > best["sharpe"]:
                        best = {"sl": sl, "tp": tp, "horizon": h, "sharpe": metrics["sharpe_ratio"]}
        with open(TUNE_FINE_DIR / f"fine_{strategy}.json", "w") as f:
            json.dump(best, f, indent=2)

def backtest_sharpe_metrics(strategy, sl, tp, h):
    return {"sharpe_ratio": np.random.rand()}
