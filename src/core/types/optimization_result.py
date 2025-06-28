
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

@dataclass
class OptimizationResult:
    """Resultado de optimización de parámetros de estrategia"""
    strategy_name: str
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_metric: str  # 'sharpe_ratio', 'total_return', etc.
    
    # Resultados de todas las combinaciones probadas
    all_results: List[Dict[str, Any]]
    
    # Estadísticas de la optimización
    total_combinations: int
    execution_time: float  # seconds
    timestamp: datetime
    data_period: Tuple[datetime, datetime]
    optimization_bounds: Dict[str, Tuple[float, float]]
    convergence_info: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.strategy_name:
            raise ValueError("Strategy name cannot be empty")
        
        if self.total_combinations <= 0:
            raise ValueError("Total combinations must be positive")
        
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        
        if self.convergence_info is None:
            self.convergence_info = {}
