from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

@dataclass
class MarketCondition:
    """Condición del mercado en un momento dado"""
    trend: str
    volatility: str
    volume: str
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    
    def __post_init__(self):
        """Validación post-inicialización"""
        valid_trends = {'bullish', 'bearish', 'sideways'}
        if self.trend not in valid_trends:
            raise ValueError(f"Invalid trend: {self.trend}. Must be one of {valid_trends}")
        
        valid_levels = {'low', 'medium', 'high'}
        if self.volatility not in valid_levels:
            raise ValueError(f"Invalid volatility: {self.volatility}. Must be one of {valid_levels}")
        
        if self.volume not in valid_levels:
            raise ValueError(f"Invalid volume: {self.volume}. Must be one of {valid_levels}")