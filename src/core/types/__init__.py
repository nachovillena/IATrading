"""Types module for trading system"""

from .trading_data import TradingData
from .signal_data import SignalData
from .metrics_data import MetricsData
from .optimization_result import OptimizationResult
from .portfolio_data import PortfolioData
from .strategy_config import StrategyConfig
from .backtest_result import BacktestResult
from .market_condition import MarketCondition
from .risk_metrics import RiskMetrics
from .trade_data import TradeData 

# Import constants from config
from ..config.config_loader import (
    SIGNAL_BUY,
    SIGNAL_HOLD, 
    SIGNAL_SELL,
    SIGNAL_NAMES,
    TIMEFRAMES,
    FOREX_PAIRS,
    MAJOR_STOCKS,
    OPTIMIZATION_METRICS,
    TRADE_STATUS,
    get_signal_constants,
    get_timeframes,
    get_forex_pairs,
    get_major_stocks,
    get_optimization_metrics,
    get_trade_status
)

__all__ = [
    # Data types
    'TradingData',
    'SignalData',
    'MetricsData',
    'OptimizationResult',
    'PortfolioData',
    'StrategyConfig',
    'BacktestResult',
    'MarketCondition',
    'RiskMetrics', 
    'TradeData', 
    
    # Constants (backward compatibility)
    'SIGNAL_BUY',
    'SIGNAL_HOLD',
    'SIGNAL_SELL',
    'SIGNAL_NAMES',
    'TIMEFRAMES',
    'FOREX_PAIRS',
    'MAJOR_STOCKS',
    'OPTIMIZATION_METRICS',
    'TRADE_STATUS',
    
    # Dynamic loaders
    'get_signal_constants',
    'get_timeframes',
    'get_forex_pairs',
    'get_major_stocks',
    'get_optimization_metrics',
    'get_trade_status'
]