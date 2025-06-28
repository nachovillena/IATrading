"""Trading signals module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.types import SignalData, TradingData
from ..core.exceptions import TradingSystemError

class SignalGenerator:
    """Generates trading signals from strategies"""
    
    def __init__(self):
        self.generated_signals = []
    
    def generate_signals_from_strategy(self, strategy, trading_data: TradingData) -> List[SignalData]:
        """Generate signals using a strategy"""
        
        if not trading_data or trading_data.data.empty:
            return []
        
        try:
            # Calculate indicators
            indicators = strategy.calculate_indicators(trading_data.data)
            
            # Generate signals
            signals = strategy.generate_signals(trading_data.data, indicators)
            
            # Convert to SignalData objects
            signal_data_list = []
            
            for i, (timestamp, signal) in enumerate(signals.items()):
                if signal != 0:  # Only store non-zero signals
                    signal_data = SignalData(
                        symbol=trading_data.symbol,
                        timeframe=trading_data.timeframe,
                        strategy=strategy.name,
                        timestamp=timestamp,
                        signal=int(signal),
                        confidence=self._calculate_confidence(signals, i),
                        price=float(trading_data.data['close'].iloc[i]),
                        params=strategy.params.copy()
                    )
                    signal_data_list.append(signal_data)
            
            return signal_data_list
            
        except Exception as e:
            raise TradingSystemError(f"Signal generation failed: {e}")
    
    def _calculate_confidence(self, signals: pd.Series, index: int) -> float:
        """Calculate signal confidence based on recent signal consistency"""
        
        # Look at last 5 signals for consistency
        lookback = min(5, index + 1)
        recent_signals = signals.iloc[max(0, index - lookback + 1):index + 1]
        
        if len(recent_signals) == 0:
            return 0.5
        
        # Calculate consistency (how many signals agree with current)
        current_signal = signals.iloc[index]
        agreement = (recent_signals == current_signal).sum() / len(recent_signals)
        
        return float(agreement)

class SignalProcessor:
    """Processes and filters trading signals"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'min_confidence': 0.6,
            'max_signals_per_hour': 5,
            'signal_timeout_minutes': 60
        }
    
    def filter_signals(self, signals: List[SignalData]) -> List[SignalData]:
        """Filter signals based on quality criteria"""
        
        filtered_signals = []
        
        for signal in signals:
            if self._should_include_signal(signal):
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _should_include_signal(self, signal: SignalData) -> bool:
        """Check if signal meets quality criteria"""
        
        # Confidence filter
        if signal.confidence < self.config['min_confidence']:
            return False
        
        # Price validation
        if signal.price <= 0:
            return False
        
        return True
    
    def aggregate_signals(self, signals: List[SignalData]) -> Dict[str, SignalData]:
        """Aggregate signals by symbol, keeping the most confident"""
        
        aggregated = {}
        
        for signal in signals:
            key = f"{signal.symbol}_{signal.timeframe}"
            
            if key not in aggregated or signal.confidence > aggregated[key].confidence:
                aggregated[key] = signal
        
        return aggregated
    
    def get_consensus_signal(self, signals: List[SignalData]) -> Optional[SignalData]:
        """Get consensus signal from multiple strategies"""
        
        if not signals:
            return None
        
        # Group by signal direction
        buy_signals = [s for s in signals if s.signal > 0]
        sell_signals = [s for s in signals if s.signal < 0]
        
        # Determine consensus
        if len(buy_signals) > len(sell_signals):
            # Buy consensus
            best_buy = max(buy_signals, key=lambda s: s.confidence)
            return best_buy
        elif len(sell_signals) > len(buy_signals):
            # Sell consensus
            best_sell = max(sell_signals, key=lambda s: s.confidence)
            return best_sell
        else:
            # No consensus or tie
            return None

class SignalManager:
    """Manages signal generation and aggregation"""
    
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.generated_signals = {}
    
    def generate_signals(self, strategy_name: str, symbol: str, timeframe: str = "H1") -> pd.DataFrame:
        """Generate signals for a specific strategy and symbol"""
        try:
            # Import strategy manager here to avoid circular imports
            from ..strategies import StrategyManager
            
            # Create strategy
            strategy_manager = StrategyManager()
            strategy = strategy_manager.create_strategy(strategy_name, {})
            
            # Get data (mock for now)
            trading_data = self._get_data(symbol, timeframe)
            
            # Generate signals
            signal_data_list = self.signal_generator.generate_signals_from_strategy(strategy, trading_data)
            
            # Convert to DataFrame
            signals_df = pd.DataFrame([
                {
                    'timestamp': signal.timestamp,
                    'signal': 'BUY' if signal.signal > 0 else 'SELL' if signal.signal < 0 else 'HOLD',
                    'confidence': signal.confidence,
                    'strategy': signal.strategy,
                    'price': signal.price
                }
                for signal in signal_data_list
            ])
            
            # Store signals
            key = f"{strategy_name}_{symbol}_{timeframe}"
            self.generated_signals[key] = signals_df
            
            return signals_df
            
        except Exception as e:
            raise TradingSystemError(f"Failed to generate signals: {e}")
    
    def aggregate_signals(self, signals_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Aggregate signals from multiple strategies"""
        if not signals_list:
            return pd.DataFrame()
        
        try:
            # Combine all signals
            combined = pd.concat(signals_list, ignore_index=True)
            
            # Group by timestamp and aggregate
            aggregated = combined.groupby('timestamp').agg({
                'signal': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'HOLD',
                'confidence': 'mean',
                'strategy': lambda x: ','.join(x.unique()),
                'price': 'first'
            }).reset_index()
            
            return aggregated
            
        except Exception as e:
            raise TradingSystemError(f"Failed to aggregate signals: {e}")
    
    def get_signals_status(self) -> Dict[str, Any]:
        """Get status of generated signals"""
        status = {}
        
        for key, signals in self.generated_signals.items():
            strategy, symbol, timeframe = key.split('_')
            
            status[strategy] = {
                'last_update': datetime.now().isoformat(),
                'total_signals': len(signals),
                'recent_signals': len(signals.tail(10)) if len(signals) > 0 else 0
            }
        
        return status
    
    def export_signals(self, strategy: str, symbol: str, output_path: str):
        """Export signals to CSV"""
        key = f"{strategy}_{symbol}_H1"  # Default timeframe
        
        if key in self.generated_signals:
            signals = self.generated_signals[key]
            signals.to_csv(output_path, index=False)
        else:
            raise TradingSystemError(f"No signals found for {strategy} on {symbol}")
    
    def _get_data(self, symbol: str, timeframe: str) -> TradingData:
        """Get trading data (mock implementation)"""
        # Create mock data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Generate realistic price data
        np.random.seed(42)
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, 100))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.0005, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, 100))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        return TradingData(
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            source="mock"
        )
