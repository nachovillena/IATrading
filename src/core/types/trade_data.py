from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

@dataclass
class TradeData:
    """Información de una operación individual"""
    trade_id: str
    symbol: str
    strategy_name: str
    entry_time: datetime
    entry_price: float
    quantity: float  # Positivo = long, negativo = short
    side: str  # 'BUY' or 'SELL'
    
    # CAMPOS OPCIONALES
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "OPEN"  # 'OPEN', 'CLOSED', 'CANCELLED'
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.trade_id:
            raise ValueError("Trade ID cannot be empty")
        
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if self.metadata is None:
            self.metadata = {}
        
        if self.side not in ['BUY', 'SELL']:
            raise ValueError("Side must be 'BUY' or 'SELL'")
    
    def calculate_pnl(self) -> Optional[float]:
        """Calcula el PnL de la operación"""
        if self.exit_price is None or self.quantity == 0:
            return None
        
        if self.quantity > 0:  # Long position
            pnl = (self.exit_price - self.entry_price) * abs(self.quantity)
        else:  # Short position
            pnl = (self.entry_price - self.exit_price) * abs(self.quantity)
        
        return pnl - self.fees
    
    def is_profitable(self) -> Optional[bool]:
        """Retorna si la operación es rentable"""
        pnl = self.calculate_pnl()
        return pnl > 0 if pnl is not None else None
