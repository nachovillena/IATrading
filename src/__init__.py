"""
AI Trading System

A comprehensive trading system with data acquisition, technical analysis,
machine learning, and strategy execution capabilities.
"""

# Core components
from .core.types import TradingData, SignalData
from .core.exceptions import TradingSystemError, DataError, StrategyError

# Data components
from .data.pipeline import DataPipeline
from .data.cache import DataCache

# Strategy components - ✅ Import actualizado
from .strategies import StrategyManager

# ML components
from .ml.features import FeatureEngineer

# Service components
from .services.orchestrator import TradingOrchestrator

# Interface components
from .interfaces.cli_interface import CLIInterface

__version__ = "1.0.0"
__author__ = "AI Trading Team"

__all__ = [
    # Core
    'TradingData',
    'SignalData', 
    'TradingSystemError',
    'DataError',
    'StrategyError',
    'DataCache',
    # Data
    'DataPipeline',
    
    # Strategies
    'StrategyManager',
    
    # ML
    'FeatureEngineer',
    
    # Services
    'TradingOrchestrator',
    
    # Interfaces
    'CLIInterface'
]
