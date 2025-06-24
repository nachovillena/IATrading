import pandas as pd
import json
from pathlib import Path
from config import Config
from utils import load_data, extract_features

PARAM_DIR = Path("parameters")
CONFIG_MASTER = Path("config_master.json")

def validate_parameters() -> None:
    master = {}
    for csv in PARAM_DIR.glob("param_grid_*.csv"):
        strategy = csv.stem.replace("param_grid_", "")
        df = pd.read_csv(csv)
        valid = []
        for _, row in df.iterrows():
            # generate signals stub
            X = extract_features(load_data(Config.load("config/config.yaml").data_paths["raw"]), strategy)
            if not X.empty:
                valid.append(row.to_dict())
        if valid:
            best = valid[:5]
            master[strategy] = best
    with open(CONFIG_MASTER, 'w') as f:
        json.dump(master, f, indent=2)
