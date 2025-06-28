"""MACD (Moving Average Convergence Divergence) strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.core.base import BaseStrategy
from src.core.types import TradingData, SignalData
from src.core.exceptions import StrategyError


class MacdStrategy(BaseStrategy):
    """MACD strategy using signal line crossovers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)
        self.context_timeframes = self.config.get('context_timeframes', ['H1'])
        
        # Validation
        if self.fast_period >= self.slow_period:
            raise StrategyError("Fast period must be less than slow period")
        
        self.logger.info(f"MACD Strategy initialized: {self.fast_period}/{self.slow_period}/{self.signal_period}")

    def calculate_indicators(self, data: TradingData) -> pd.DataFrame:
        """Calculate MACD indicators"""
        if data.data.empty:
            raise StrategyError("Cannot calculate indicators on empty data")
        
        # CORREGIDO: Acceso correcto a los datos del DataFrame
        df = data.data.copy()  # data.data es el DataFrame, no data directamente
        
        # Calculate MACD components
        ema_fast = df['close'].ewm(span=self.fast_period).mean()
        ema_slow = df['close'].ewm(span=self.slow_period).mean()
        
        # MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        df['macd_signal'] = df['macd'].ewm(span=self.signal_period).mean()
        
        # MACD histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df

    def generate_signals(self, data: TradingData, context_data: Optional[Dict[str, TradingData]] = None) -> SignalData:
        """
        Generate MACD trading signals, usando contexto multi-timeframe si está disponible.
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
                    # Calcula MACD en la temporalidad superior
                    ema_fast = tf_df['close'].ewm(span=self.fast_period).mean()
                    ema_slow = tf_df['close'].ewm(span=self.slow_period).mean()
                    tf_df['macd'] = ema_fast - ema_slow
                    tf_df['macd_signal'] = tf_df['macd'].ewm(span=self.signal_period).mean()
                    tf_df['trend'] = np.where(tf_df['macd'] > tf_df['macd_signal'], 1, -1)
                    # Reindexar a la temporalidad principal (forward fill)
                    reindexed_trend = tf_df['trend'].reindex(df.index, method='ffill').fillna(0)
                    # Solo permitimos señales si la tendencia superior es igual a la local
                    local_trend = np.where(df['macd'] > df['macd_signal'], 1, -1)
                    higher_trend_ok &= (reindexed_trend == local_trend)

            # --- Señales base MACD crossover ---
            signals = pd.Series(0, index=df.index)

            # Buy: cruce alcista + tendencia superior compatible
            buy_condition = (
                (df['macd'] > df['macd_signal']) &
                (df['macd'].shift() <= df['macd_signal'].shift()) &
                higher_trend_ok
            )
            signals[buy_condition] = 1

            # Sell: cruce bajista + tendencia superior compatible
            sell_condition = (
                (df['macd'] < df['macd_signal']) &
                (df['macd'].shift() >= df['macd_signal'].shift()) &
                higher_trend_ok
            )
            signals[sell_condition] = -1

            # Mantener posición (relleno hacia adelante)
            signals = signals.replace(0, np.nan).ffill().fillna(0)

            # Metadatos
            current_macd = df['macd'].iloc[-1] if not df['macd'].empty else 0
            current_signal = df['macd_signal'].iloc[-1] if not df['macd_signal'].empty else 0
            last_signal = signals.iloc[-1] if not signals.empty else 0
            signal_changes = (signals != signals.shift()).sum()

            metadata = {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'current_macd': round(current_macd, 6),
                'current_signal': round(current_signal, 6),
                'last_signal': int(last_signal),
                'signal_changes': int(signal_changes),
                'trend': 'bullish' if current_macd > current_signal else 'bearish',
                'context_timeframes': list(context_data.keys()) if context_data else ['H1']
            }

            self.logger.info(f"MACD signals generated: {signal_changes} changes, current MACD: {current_macd:.4f}")

            return SignalData(
                signals=signals,
                metadata=metadata,
                strategy_name='macd',
                timestamp=pd.Timestamp.now()
            )

        except Exception as e:
            self.logger.error(f"Error generating MACD signals: {e}")
            raise StrategyError(f"MACD signal generation failed: {e}")

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for optimization"""
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """Alias for get_default_parameters for backward compatibility"""
        return self.get_default_parameters()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'MACD',
            'type': 'momentum',
            'description': 'MACD signal line crossover strategy',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period
            },
            'signals': {  # AGREGADO: Información sobre las señales
                'buy': 'MACD line crosses above signal line',
                'sell': 'MACD line crosses below signal line',
                'histogram': 'Shows momentum strength'
            },
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'markets': ['forex', 'stocks', 'crypto']
        }