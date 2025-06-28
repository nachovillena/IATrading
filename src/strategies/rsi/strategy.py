"""RSI (Relative Strength Index) strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.core.base import BaseStrategy
from src.core.types import TradingData, SignalData
from src.core.exceptions import StrategyError

class RsiStrategy(BaseStrategy):
    """RSI strategy using overbought/oversold levels"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.oversold = self.config.get('oversold', 30)
        self.overbought = self.config.get('overbought', 70)
        self.context_timeframes = self.config.get('context_timeframes', ['H1'])
        
        # Validation
        if not (0 < self.oversold < self.overbought < 100):
            raise StrategyError("Invalid RSI thresholds: oversold < overbought and both between 0-100")
        
        self.logger.info(f"RSI Strategy initialized: period={self.rsi_period}, levels={self.oversold}/{self.overbought}")

    def calculate_indicators(self, data: TradingData) -> pd.DataFrame:
        """Calculate RSI indicator"""
        if data.data.empty:
            raise StrategyError("Cannot calculate indicators on empty data")
        
        df = data.data.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def generate_signals(self, data: TradingData, context_data: Optional[Dict[str, TradingData]] = None) -> SignalData:
        """
        Generate RSI trading signals, usando contexto multi-timeframe si está disponible.
        Solo se activa una señal si la tendencia en temporalidades superiores es compatible.
        """
        try:
            df = self.calculate_indicators(data)

            # --- Lógica multi-timeframe ---
            higher_trend_ok = pd.Series(True, index=df.index)
            if context_data is not None and len(context_data) > 1:
                for tf, tf_td in context_data.items():
                    tf_df = tf_td.data  # <-- Accede al DataFrame interno
                    if tf_df is df:
                        continue
                    tf_df = tf_df.copy()
                    # Calcula RSI en la temporalidad superior
                    delta = tf_df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                    rs = gain / loss
                    tf_df['rsi'] = 100 - (100 / (1 + rs))
                    # Considera tendencia alcista si RSI > 50, bajista si < 50
                    tf_df['trend'] = np.where(tf_df['rsi'] > 50, 1, -1)
                    # Reindexar a la temporalidad principal (forward fill)
                    reindexed_trend = tf_df['trend'].reindex(df.index, method='ffill').fillna(0)
                    # Solo permitimos señales si la tendencia superior es igual a la local
                    local_trend = np.where(df['rsi'] > 50, 1, -1)
                    higher_trend_ok &= (reindexed_trend == local_trend)

            # --- Señales base RSI ---
            signals = pd.Series(0, index=df.index)

            # Buy: cruce alcista sobre nivel de sobreventa + tendencia superior compatible
            buy_condition = (
                (df['rsi'] > self.oversold) &
                (df['rsi'].shift() <= self.oversold) &
                higher_trend_ok
            )
            signals[buy_condition] = 1

            # Sell: cruce bajista bajo nivel de sobrecompra + tendencia superior compatible
            sell_condition = (
                (df['rsi'] < self.overbought) &
                (df['rsi'].shift() >= self.overbought) &
                higher_trend_ok
            )
            signals[sell_condition] = -1

            # Mantener posición (relleno hacia adelante)
            signals = signals.replace(0, np.nan).ffill().fillna(0)

            # Metadatos
            current_rsi = df['rsi'].iloc[-1] if not df['rsi'].empty else 50
            last_signal = signals.iloc[-1] if not signals.empty else 0
            signal_changes = (signals != signals.shift()).sum()

            metadata = {
                'rsi_period': self.rsi_period,
                'oversold_threshold': self.oversold,
                'overbought_threshold': self.overbought,
                'current_rsi': round(current_rsi, 2),
                'last_signal': int(last_signal),
                'signal_changes': int(signal_changes),
                'market_condition': self._determine_market_condition(current_rsi),
                'context_timeframes': list(context_data.keys()) if context_data else ['H1']
            }

            self.logger.info(f"RSI signals generated: {signal_changes} changes, current RSI: {current_rsi:.2f}")

            return SignalData(
                signals=signals,
                metadata=metadata,
                strategy_name='rsi',
                timestamp=pd.Timestamp.now()
            )

        except Exception as e:
            self.logger.error(f"Error generating RSI signals: {e}")
            raise StrategyError(f"RSI signal generation failed: {e}")

    def _determine_market_condition(self, rsi: float) -> str:
        """Determine market condition based on RSI"""
        if rsi >= self.overbought:
            return 'overbought'
        elif rsi <= self.oversold:
            return 'oversold'
        else:
            return 'neutral'

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for optimization"""
        return {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70
        }
    
    # AGREGADO: Método que esperan los tests
    def get_default_params(self) -> Dict[str, Any]:
        """Alias for get_default_parameters for backward compatibility"""
        return self.get_default_parameters()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'RSI',
            'type': 'momentum',
            'description': 'Relative Strength Index overbought/oversold strategy',
            'parameters': {
                'rsi_period': self.rsi_period,
                'oversold': self.oversold,
                'overbought': self.overbought
            },
            'signals': {  # AGREGADO: Información sobre las señales
                'buy': f'RSI crosses above {self.oversold} (oversold level)',
                'sell': f'RSI crosses below {self.overbought} (overbought level)',
                'neutral': f'RSI between {self.oversold}-{self.overbought}'
            },
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'markets': ['forex', 'stocks', 'crypto']
        }