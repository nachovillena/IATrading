"""Optimization service using Optuna"""

import optuna
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from ..core.types import OptimizationResult
from ..ml.optimizer import StrategyOptimizer
from .config_service import ConfigService

class OptimizationService:
    """Service for strategy optimization"""
    
    def __init__(self, config_service: ConfigService):
        self.config = config_service
        self.optimizer = StrategyOptimizer()
        self.cache = {}  # Simple in-memory cache for optimization results

    def optimize_strategy(self, strategy: str, symbol: str, 
                         data: pd.DataFrame) -> OptimizationResult:
        """Optimize strategy parameters"""
        
        try:
            start_time = datetime.now()
            
            # Get optimization space from strategy
            param_space = self.optimizer.get_parameter_space(strategy)
            
            # Run optimization
            best_params, best_score = self.optimizer.optimize(
                strategy=strategy,
                data=data,
                param_space=param_space,
                n_trials=self.config.get_optimization_config().get('trials', 100)
            )

            # Create and cache result
            result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_time=(datetime.now() - start_time).total_seconds(),
                strategy=strategy,
                timestamp=datetime.now()
            )
            
            self.cache[(symbol, strategy)] = result
            return result

        except Exception as e:
            raise RuntimeError(f"Optimization failed for {strategy}: {str(e)}")

    def get_best_parameters(self, symbol: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Get cached optimization results"""
        result = self.cache.get((symbol, strategy))
        return result.best_params if result else None
