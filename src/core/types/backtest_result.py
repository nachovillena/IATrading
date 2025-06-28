from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

from .metrics_data import MetricsData
from .signal_data import SignalData

@dataclass
class BacktestResult:
    """Resultado de un backtest"""
    strategy_name: str
    metrics: MetricsData
    signals: SignalData
    equity_curve: pd.Series
    trades: pd.DataFrame
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.strategy_name:
            raise ValueError("Strategy name cannot be empty")