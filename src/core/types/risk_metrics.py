from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

@dataclass
class RiskMetrics:
    """Métricas de riesgo para trading"""
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_per_trade: float = 0.02  # 2% default
    max_positions: int = 3
    correlation_limit: float = 0.8
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1")
        
        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")