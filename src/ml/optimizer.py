from typing import Dict, Any, Tuple
import optuna
import pandas as pd
from datetime import datetime

from ..core.exceptions import OptimizationError
from ..core.types import OptimizationResult

class StrategyOptimizer:
    """Optimizador de parámetros para estrategias de trading usando Optuna"""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
    
    def optimize(self, strategy_class, data: pd.DataFrame, 
                param_space: Dict[str, Tuple], **kwargs) -> OptimizationResult:
        """
        Optimiza los parámetros de una estrategia usando Optuna
        
        Args:
            strategy_class: Clase de la estrategia a optimizar
            data: DataFrame con datos históricos
            param_space: Diccionario con rangos de parámetros a optimizar
            **kwargs: Argumentos adicionales para la optimización
        """
        try:
            study = optuna.create_study(direction="maximize")
            
            def objective(trial):
                # Crear parámetros de prueba
                trial_params = {}
                for param, (min_val, max_val) in param_space.items():
                    if isinstance(min_val, int):
                        trial_params[param] = trial.suggest_int(param, min_val, max_val)
                    else:
                        trial_params[param] = trial.suggest_float(param, min_val, max_val)
                
                # Evaluar estrategia con estos parámetros
                strategy = strategy_class(trial_params)
                return self._evaluate_strategy(strategy, data)
            
            # Ejecutar optimización
            study.optimize(objective, n_trials=self.n_trials)
            
            # Crear resultado
            result = OptimizationResult(
                best_params=study.best_params,
                best_score=study.best_value,
                optimization_time=study.duration,
                strategy=strategy_class.__name__,
                symbol=kwargs.get('symbol', 'unknown'),
                timeframe=kwargs.get('timeframe', 'unknown'),
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            raise OptimizationError(f"Error optimizing strategy: {str(e)}")
    
    def _evaluate_strategy(self, strategy, data: pd.DataFrame) -> float:
        """Evalúa el rendimiento de una estrategia"""
        try:
            # Calcular indicadores
            indicators = strategy.calculate_indicators(data)
            
            # Generar señales
            signals = strategy.generate_signals(data, indicators)
            
            # Calcular retorno
            returns = self._calculate_returns(data, signals)
            
            # Calcular Sharpe Ratio como métrica de optimización
            return self._calculate_sharpe_ratio(returns)
            
        except Exception as e:
            raise OptimizationError(f"Error evaluating strategy: {str(e)}")
    
    def _calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calcula los retornos de la estrategia"""
        price_changes = data['close'].pct_change()
        return signals.shift(1) * price_changes
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcula el Sharpe Ratio de los retornos"""
        if returns.empty or returns.isna().all():
            return -999999  # Penalización para parámetros inválidos
            
        annual_factor = 252  # Días de trading en un año
        return (returns.mean() * annual_factor) / (returns.std() * (annual_factor ** 0.5))