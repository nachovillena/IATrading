"""Base analyzer class"""
from ...utils.logger import Logger
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..types import TradingData


class BaseAnalyzer(ABC):
    """Base class for data analyzers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, data: TradingData) -> Dict[str, Any]:
        """Analyze trading data"""
        pass
    
    def validate_data(self, data: TradingData) -> bool:
        """Validate input data"""
        return data is not None and not data.data.empty