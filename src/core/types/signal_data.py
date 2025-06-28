"""Signal data type for trading signals"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from ...utils.logger import Logger


@dataclass
class SignalData:
    """Container for trading signals and metadata"""
    
    signals: pd.Series
    metadata: Dict[str, Any]
    confidence: float = 0.0
    strategy_name: str = ""
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}
        
        # Validate signals
        if self.signals is not None and not self.signals.empty:
            # Ensure signals are in valid range (-1, 0, 1)
            valid_signals = self.signals.isin([-1, 0, 1])
            if not valid_signals.all():
                # Log warning but don't fail
                
                class_name = self.__class__.__name__
                self.logger = Logger(
                    name=class_name,
                    to_console=True,
                    to_file=f"{class_name}.log"
                )
                self.name = class_name
                self.logger.warning(f"Invalid signal values found: {self.signals[~valid_signals].unique()}")
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary statistics of signals"""
        if self.signals is None or self.signals.empty:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'signal_changes': 0
            }
        
        signal_counts = self.signals.value_counts()
        signal_changes = (self.signals != self.signals.shift()).sum()
        
        return {
            'total_signals': len(self.signals),
            'buy_signals': signal_counts.get(1, 0),
            'sell_signals': signal_counts.get(-1, 0),
            'hold_signals': signal_counts.get(0, 0),
            'signal_changes': signal_changes,
            'last_signal': self.signals.iloc[-1] if not self.signals.empty else 0
        }
    
    def copy(self):
        """Create a copy of the SignalData"""
        return SignalData(
            signals=self.signals.copy(),
            metadata=self.metadata.copy(),
            confidence=self.confidence,
            strategy_name=self.strategy_name,
            timestamp=self.timestamp
        )