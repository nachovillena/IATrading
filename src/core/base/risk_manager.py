"""Base risk manager class"""


from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..types import SignalData
from ...utils.logger import Logger


class BaseRiskManager(ABC):
    """Base class for risk management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.20)  # 20%
        self.max_positions = self.config.get('max_positions', 5)
    
    @abstractmethod
    def calculate_position_size(self, signal: SignalData, account_balance: float) -> float:
        """Calculate position size based on risk parameters"""
        pass
    
    @abstractmethod
    def validate_trade(self, signal: SignalData, current_positions: List[Dict]) -> bool:
        """Validate if a trade should be executed"""
        pass
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits"""
        return {
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_positions': self.max_positions
        }