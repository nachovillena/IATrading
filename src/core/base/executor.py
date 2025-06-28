"""Base executor class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..types import SignalData
from ...utils.logger import Logger


class BaseExecutor(ABC):
    """Base class for trade execution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.is_connected = False
    
    @abstractmethod
    def execute_trade(self, signal: SignalData, position_size: float) -> bool:
        """Execute a trade based on signal"""
        pass
    
    @abstractmethod
    def close_position(self, position_id: str) -> bool:
        """Close an existing position"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    def connect(self) -> bool:
        """Connect to trading platform"""
        # Default implementation
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from trading platform"""
        # Default implementation
        self.is_connected = False
        return True