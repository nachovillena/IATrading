"""Base optimizer class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import optuna

from ..types import TradingData
from .strategy import BaseStrategy
from ...utils.logger import Logger

class BaseOptimizer(ABC):
    """Base class for parameter optimization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.optimization_metric = self.config.get('optimization_metric', 'sharpe')
        self.n_trials = self.config.get('n_trials', 50)
        self.direction = self.config.get('direction', 'maximize')
    
    @abstractmethod
    def optimize(self, strategy: BaseStrategy, data: TradingData, 
                 param_ranges: Dict[str, tuple]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        pass
    
    def get_optimization_bounds(self, strategy: BaseStrategy) -> Dict[str, tuple]:
        """Get default optimization bounds for strategy"""
        # Puedes obtener los bounds del param_space del JSON o del método de la estrategia
        if hasattr(strategy, "get_param_space"):
            param_space = strategy.get_param_space()
            # Convierte listas a tuplas para Optuna
            return {k: tuple(v) if isinstance(v, list) and len(v) == 2 else v for k, v in param_space.items()}
        return {}

class OptunaOptimizer(BaseOptimizer):
    """Optuna-based optimizer for trading strategies"""

    def optimize(self, strategy: BaseStrategy, data: TradingData, 
                 param_ranges: Dict[str, tuple]) -> Dict[str, Any]:
        """
        Optimiza los parámetros de la estrategia usando Optuna.
        """
        def objective(trial):
            params = {}
            for param, bounds in param_ranges.items():
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    # Rango numérico
                    if all(isinstance(x, int) for x in bounds):
                        params[param] = trial.suggest_int(param, bounds[0], bounds[1])
                    else:
                        params[param] = trial.suggest_float(param, bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    params[param] = trial.suggest_categorical(param, bounds)
                else:
                    params[param] = bounds

            # Validación de parámetros
            try:
                strategy.validate_parameters(params)
            except Exception as e:
                self.logger.warning(f"Invalid params: {params} ({e})")
                raise optuna.TrialPruned()

            # Generar señales y backtest
            try:
                strategy_instance = type(strategy)(config={**strategy.config, **params})
                signals = strategy_instance.generate_signals(data)
                bt_result = strategy_instance.backtest_simple(
                    main_data=data.data,
                    signals=signals.signals,
                    commission=params.get('commission', 0.0),
                    slippage=params.get('slippage', 0.0),
                    stop_loss=params.get('stop_loss', None),
                    take_profit=params.get('take_profit', None)
                )
                metric = bt_result.get(self.optimization_metric, 0)
                if metric is None or (self.direction == "maximize" and metric < 0) or (self.direction == "minimize" and metric > 0):
                    raise optuna.TrialPruned()
                return metric
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                raise optuna.TrialPruned()

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_value = study.best_value

        self.logger.info(f"Best params: {best_params} | Best {self.optimization_metric}: {best_value}")

        return {
            "best_params": best_params,
            "best_score": best_value,
            "study": study
        }