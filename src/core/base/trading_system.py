"""Base trading system class"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from .provider import BaseProvider
from .strategy import BaseStrategy
from .risk_manager import BaseRiskManager
from .executor import BaseExecutor
from ...utils.logger import Logger


class BaseTradingSystem:
    """Base trading system that coordinates all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        
        # Components
        self.provider: Optional[BaseProvider] = None
        self.strategies: List[BaseStrategy] = []
        self.risk_manager: Optional[BaseRiskManager] = None
        self.executor: Optional[BaseExecutor] = None
        
        # Status
        self.is_running = False
        self.last_update = None
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the system"""
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.name}")
    
    def set_provider(self, provider: BaseProvider) -> None:
        """Set data provider"""
        self.provider = provider
        self.logger.info(f"Set provider: {provider.name}")
    
    def set_risk_manager(self, risk_manager: BaseRiskManager) -> None:
        """Set risk manager"""
        self.risk_manager = risk_manager
        self.logger.info("Risk manager configured")
    
    def set_executor(self, executor: BaseExecutor) -> None:
        """Set trade executor"""
        self.executor = executor
        self.logger.info("Trade executor configured")
    
    def start(self) -> bool:
        """Start the trading system"""
        if not self.validate_system():
            return False
        
        self.is_running = True
        self.last_update = datetime.now()
        self.logger.info("Trading system started")
        return True
    
    def stop(self) -> bool:
        """Stop the trading system"""
        self.is_running = False
        self.logger.info("Trading system stopped")
        return True
    
    def validate_system(self) -> bool:
        """Validate system configuration"""
        if not self.provider:
            self.logger.error("No data provider configured")
            return False
        
        if not self.strategies:
            self.logger.error("No strategies configured")
            return False
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'provider': self.provider.name if self.provider else None,
            'strategies': [s.name for s in self.strategies],
            'risk_manager': self.risk_manager is not None,
            'executor': self.executor is not None
        }