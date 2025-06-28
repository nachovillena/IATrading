"""MetricsData type definition"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MetricsData:
    """Contenedor para métricas de trading y backtesting"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Métricas adicionales
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    
    # Metadata
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    strategy_name: str = ""
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if self.total_trades < 0:
            raise ValueError("Total trades cannot be negative")
        
        if not 0 <= self.win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")