"""Core module for trading system"""

# Import types from new structure (when available)
try:
    from .types import *
except ImportError:
    # Fallback to old structure during migration
    try:
        from .types import (
            TradingData,
            SignalData,
            MetricsData,
            OptimizationResult,
            PortfolioData,
            StrategyConfig,
            BacktestResult,
            MarketCondition,
            RiskMetrics,
            TradeData,
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
    except ImportError as e:
        print(f"Warning: Could not import types: {e}")
        # Create minimal fallback types
        from dataclasses import dataclass
        from typing import Any, Dict
        import pandas as pd
        from datetime import datetime
        
        @dataclass
        class TradingData:
            symbol: str = ""
            timeframe: str = ""
            data: pd.DataFrame = None
            provider: str = ""
            timestamp: datetime = None
            quality_score: float = 0.0
        
        @dataclass
        class SignalData:
            signals: pd.Series = None
            metadata: Dict[str, Any] = None
            confidence: float = 0.0
            strategy_name: str = ""

# Import exceptions with fallbacks
try:
    from .exceptions import (
        TradingSystemError,
        DataError,
        StrategyError,
        ValidationError,
        ConfigurationError,
        MLModelError
    )
except ImportError:
    # Create minimal exception classes if they don't exist
    class TradingSystemError(Exception):
        """Base exception for trading system errors"""
        pass
    
    class DataError(TradingSystemError):
        """Exception raised for data-related errors"""
        pass
    
    class StrategyError(TradingSystemError):
        """Exception raised for strategy-related errors"""
        pass
    
    class ValidationError(TradingSystemError):
        """Exception raised for validation errors"""
        pass
    
    class ConfigurationError(TradingSystemError):
        """Exception raised for configuration errors"""
        pass

# Import base classes with fallbacks
try:
    from .base import (
        BaseProvider,
        BaseStrategy,
        BaseAnalyzer,
        BaseRiskManager,
        BaseExecutor,
        BaseBacktester,
        BaseOptimizer,
        BaseTradingSystem
    )
except ImportError as e:
    print(f"Warning: Could not import base classes: {e}")
    # Create minimal base classes
    from abc import ABC, abstractmethod
    
    class BaseProvider(ABC):
        pass
    
    class BaseStrategy(ABC):
        pass
    
    class BaseAnalyzer(ABC):
        pass
    
    class BaseRiskManager(ABC):
        pass
    
    class BaseExecutor(ABC):
        pass
    
    class BaseBacktester(ABC):
        pass
    
    class BaseOptimizer(ABC):
        pass
    
    class BaseTradingSystem:
        pass

__all__ = [
    # Types
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
    
    # Constants
    'SIGNAL_BUY',
    'SIGNAL_HOLD',
    'SIGNAL_SELL',
    'SIGNAL_NAMES',
    'TIMEFRAMES',
    'FOREX_PAIRS',
    'MAJOR_STOCKS',
    'OPTIMIZATION_METRICS',
    'TRADE_STATUS',
    
    # Exceptions
    'TradingSystemError',
    'DataError',
    'StrategyError',
    'ValidationError',
    'ConfigurationError',
    
    # Base classes
    'BaseProvider',
    'BaseStrategy',
    'BaseAnalyzer',
    'BaseRiskManager',
    'BaseExecutor',
    'BaseBacktester',
    'BaseOptimizer',
    'BaseTradingSystem'
]
