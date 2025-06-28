"""Trading data type for OHLCV data"""

from dataclasses import dataclass
from datetime import datetime
import traceback
from typing import Optional
import pandas as pd
from ...utils.logger import Logger


@dataclass
class TradingData:
    """Container for trading OHLCV data"""
    
    symbol: str
    timeframe: str
    data: pd.DataFrame
    provider: str
    timestamp: datetime
    quality_score: float = 0.0
    
    def __post_init__(self):
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        """Post-initialization validation"""
        self.logger.debug(f"[DEBUG] TradingData creado: symbol={self.symbol}, timeframe={self.timeframe}, provider={self.provider}, timestamp={self.timestamp}")
        if self.symbol is None or self.timeframe is None or self.data is None or self.provider is None or self.timestamp is None:
            traceback.print_stack()
        if self.data is None:
            self.data = pd.DataFrame()
        
        # Ensure timestamp is set
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validate data structure
        if not self.data.empty:
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns in {self.symbol}: {missing_cols}")
    
    def copy(self):
        """Create a copy of the TradingData"""
        return TradingData(
            symbol=self.symbol,
            timeframe=self.timeframe,
            data=self.data.copy(),
            provider=self.provider,
            timestamp=self.timestamp,
            quality_score=self.quality_score
        )
    
    def get_latest_price(self) -> Optional[float]:
        """Get the latest close price"""
        if self.data.empty or 'close' not in self.data.columns:
            return None
        return self.data['close'].iloc[-1]
    
    def get_price_range(self) -> Optional[tuple]:
        """Get (min, max) price range"""
        if self.data.empty or 'low' not in self.data.columns or 'high' not in self.data.columns:
            return None
        return (self.data['low'].min(), self.data['high'].max())
    
    def validate_data(self) -> bool:
        """Validate data integrity"""
        if self.data.empty:
            return False
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            return False
        
        # Check for logical consistency
        invalid_data = (
            (self.data['high'] < self.data['low']) |
            (self.data['open'] < 0) | (self.data['high'] < 0) |
            (self.data['low'] < 0) | (self.data['close'] < 0)
        ).any()
        
        return not invalid_data