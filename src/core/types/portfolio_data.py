from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

@dataclass
class PortfolioData:
    """Contenedor para datos de portfolio y gestión de múltiples posiciones"""
    # ARGUMENTOS REQUERIDOS PRIMERO
    portfolio_id: str
    positions: Dict[str, Dict[str, Any]]  # symbol -> position info
    cash_balance: float
    total_value: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    created_at: datetime
    last_updated: datetime
    strategy_allocations: Dict[str, float]  # strategy -> allocation %
    
    # ARGUMENTOS OPCIONALES AL FINAL
    max_position_size: float = 0.1  # 10% max por posición
    max_correlation: float = 0.8
    max_sector_exposure: float = 0.3  # 30% max por sector
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.portfolio_id:
            raise ValueError("Portfolio ID cannot be empty")
        
        if self.cash_balance < 0:
            raise ValueError("Cash balance cannot be negative")
        
        if self.total_value < 0:
            raise ValueError("Total value cannot be negative")
        
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
    
    def get_position_count(self) -> int:
        """Retorna el número de posiciones activas"""
        return len([pos for pos in self.positions.values() if pos.get('quantity', 0) != 0])
    
    def get_total_exposure(self) -> float:
        """Retorna la exposición total del portfolio"""
        total_exposure = sum(
            abs(pos.get('quantity', 0) * pos.get('price', 0))
            for pos in self.positions.values()
        )
        return total_exposure
    
    def get_cash_utilization(self) -> float:
        """Retorna el porcentaje de cash utilizado"""
        if self.total_value == 0:
            return 0.0
        return (self.total_value - self.cash_balance) / self.total_value