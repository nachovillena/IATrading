import pandas as pd
import optuna
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict
from utils import load_data, backtest_signals
from config import Config

PARAM_DIR = Path("parameters")
PARAM_DIR.mkdir(exist_ok=True)

"""
Optimiza una estrategia de trading utilizando búsqueda bayesiana con Optuna.

Carga los datos históricos (.parquet) para un símbolo dado, define el espacio de
parámetros desde la configuración, y evalúa cada combinación con un backtest.
Guarda los mejores resultados en un CSV y muestra progreso en consola.
"""

def optimize_strategy(strategy: str, cfg: Config, symbol: str) -> None:
    """
    Ejemplo de uso:
        optimize_strategy("EMA_20_50", cfg, "EURUSD")
    """
    print(f"\n[INICIO] Optimización de estrategia: {strategy} para símbolo {symbol}")
    data_path = Path(f"data/cache/{symbol}_M1_{pd.Timestamp.now().year}.parquet")
    print(f"Cargando datos desde: {data_path}")
    data = load_data(str(data_path))
    print(f"Datos cargados: {len(data)} filas\n")

    space = cfg.parameters.get(strategy, {})
    if not space:
        print(f"[AVISO] No se encontraron parámetros definidos para la estrategia '{strategy}'")
        return

    def objective(trial: optuna.Trial) -> float:
        print(f"\n[Trial {trial.number}] Generando parámetros...")
        params: Dict[str, Any] = {}
        for k, v in space.items():
            if isinstance(v, list) and len(v) == 2:
                params[k] = trial.suggest_uniform(k, v[0], v[1])
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                params[k] = trial.suggest_int(k, int(v[0]), int(v[1]))
            else:
                raise ValueError(f"Parámetro {k} mal definido en configuración")
        print(f"Parámetros propuestos: {params}")

        signals = backtest_signals(data, strategy, params)
        metrics = compute_metrics(signals)
        print(f"  → Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        return metrics['sharpe_ratio']

    study = optuna.create_study(direction='maximize')
    print("[PROGRESO] Iniciando optimización con Optuna (50 trials)...")
    for _ in tqdm(range(50), desc="Optimizando", ncols=70):
        study.optimize(objective, n_trials=1, catch=(Exception,))

    print("\n[FIN] Optimización completada")
    trials_df = study.trials_dataframe(attrs=("params", "value"))
    output_file = PARAM_DIR / f"param_grid_{strategy}_{symbol}.csv"
    trials_df.to_csv(output_file, index=False)
    print(f"[RESULTADOS] Parámetros guardados en: {output_file}")

    # Mostrar mejor conjunto de parámetros
    best = study.best_trial
    print(f"\n[MEJOR] Trial {best.number} → Valor: {best.value:.4f}")
    print(f"        Parámetros óptimos: {best.params}\n")

def compute_metrics(signals: pd.DataFrame) -> Dict[str, Any]:
    """
    Ejemplo de uso interno:
        metrics = compute_metrics(signals)
    """
    std = signals['returns'].std()
    sharpe = signals['returns'].mean() / std if std != 0 else 0.0
    return {'sharpe_ratio': sharpe}
