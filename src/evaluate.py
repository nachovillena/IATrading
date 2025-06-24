import json
import pandas as pd
from pathlib import Path
from utils import load_equity_curve, compute_performance_metrics

eval_dir = Path("evaluations")
eval_dir.mkdir(exist_ok=True)

def evaluate_performance() -> None:
    for strategy, params in json.load(open("config_master.json")).items():
        eq = load_equity_curve(strategy)
        metrics = compute_performance_metrics(eq)
        eq.to_csv(eval_dir / f"equity_{strategy}.csv")
        with open(eval_dir / f"metrics_{strategy}.json", "w") as f:
            json.dump(metrics, f, indent=2)
