"""Strategies module for trading system"""

from .manager import StrategyManager

# Import specific strategies
from .ema.strategy import EmaStrategy
from .rsi.strategy import RsiStrategy
from .macd.strategy import MacdStrategy

__all__ = [
    'StrategyManager',
    'EmaStrategy',
    'RsiStrategy',
    'MacdStrategy'
]