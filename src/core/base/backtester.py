"""Base backtester class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..types import TradingData
from .strategy import BaseStrategy
from ...utils.logger import Logger


class BaseBacktester(ABC):
    """Base class for backtesting"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.initial_capital = self.config.get('initial_capital', 10000)
        self.commission = self.config.get('commission', 0.001)  # 0.1%
    
    @abstractmethod
    def run_backtest(self, strategy: BaseStrategy, data: TradingData) -> Dict[str, Any]:
        """Run backtest on strategy with data"""
        pass
    
    @abstractmethod
    def calculate_metrics(self, equity_curve: List[float]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        pass
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtest configuration"""
        return {
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            **self.config
        }