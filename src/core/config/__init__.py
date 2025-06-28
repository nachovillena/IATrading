"""Configuration module for trading system"""

from .config_loader import (
    ConfigLoader,
    get_signal_constants,
    get_timeframes,
    get_forex_pairs,
    get_major_stocks,
    get_optimization_metrics,
    get_trade_status,
    SIGNAL_BUY,
    SIGNAL_HOLD,
    SIGNAL_SELL,
    SIGNAL_NAMES,
    TIMEFRAMES,
    FOREX_PAIRS,
    MAJOR_STOCKS,
    OPTIMIZATION_METRICS,
    TRADE_STATUS
)

__all__ = [
    # Main config loader class
    'ConfigLoader',
    
    # Dynamic getters
    'get_signal_constants',
    'get_timeframes',
    'get_forex_pairs',
    'get_major_stocks',
    'get_optimization_metrics',
    'get_trade_status',
    
    # Constants (for backward compatibility)
    'SIGNAL_BUY',
    'SIGNAL_HOLD',
    'SIGNAL_SELL',
    'SIGNAL_NAMES',
    'TIMEFRAMES',
    'FOREX_PAIRS',
    'MAJOR_STOCKS',
    'OPTIMIZATION_METRICS',
    'TRADE_STATUS'
]