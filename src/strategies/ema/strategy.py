"""EMA (Exponential Moving Average) crossover strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.core.base import BaseStrategy
from src.core.types import TradingData, SignalData
from src.core.exceptions import StrategyError


class EmaStrategy(BaseStrategy):
    """EMA crossover strategy using fast and slow exponential moving averages"""
    
    def __init__(self, config: Dict[str, Any] = None):      
        super().__init__(config)
        self.ema_fast = self.config.get('ema_fast', 12)
        self.ema_slow = self.config.get('ema_slow', 26)
        self.signal_threshold = self.config.get('signal_threshold', 0.001)
        self.context_timeframes = self.config.get('context_timeframes', ['H1'])
        self.stop_loss = self.config.get('stop_loss', 0.015)
        self.take_profit = self.config.get('take_profit', 0.03)
        self.context_alignment = self.config.get('context_alignment', 'majority')  # 'majority', 'any', 'all'
        self.keep_position = self.config.get('keep_position', False)

    def calculate_indicators(self, data: TradingData) -> pd.DataFrame:
        """Calculate EMA indicators"""
        df = data.data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()
        
        # Calculate EMA difference percentage
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_pct'] = df['ema_diff'] / df['ema_slow']
        
        return df

    def generate_signals(self, data: TradingData, context_data: Optional[Dict[str, TradingData]] = None) -> SignalData:
        """
        Generate EMA crossover signals, usando contexto multi-timeframe si está disponible.
        La señal solo se activa si la mayoría de las temporalidades superiores están alineadas.
        """
        self.validate_data(data)
        processed_data = self.preprocess_data(data)
        df = self.calculate_indicators(processed_data)

        higher_trend_ok = pd.Series(True, index=df.index)

        # --- Contexto multi-timeframe: mayoría alineada ---
        if context_data:
            trend_matches = []
            for tf, tf_td in context_data.items():
                tf_df = tf_td.data
                tf_df = tf_df.copy()
                tf_df['ema_fast'] = tf_df['close'].ewm(span=self.ema_fast).mean()
                tf_df['ema_slow'] = tf_df['close'].ewm(span=self.ema_slow).mean()
                tf_df['trend'] = np.where(tf_df['ema_fast'] > tf_df['ema_slow'], 1, -1)
                reindexed_trend = tf_df['trend'].reindex(df.index, method='ffill').fillna(0)
                local_trend = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
                trend_matches.append(reindexed_trend == local_trend)

            if trend_matches:
                trend_matches = np.stack(trend_matches)
                if self.context_alignment == 'all':
                    higher_trend_ok = (trend_matches.sum(axis=0) == len(trend_matches))
                elif self.context_alignment == 'any':
                    higher_trend_ok = (trend_matches.sum(axis=0) >= 1)
                else:  # majority
                    higher_trend_ok = (trend_matches.sum(axis=0) >= (len(trend_matches) // 2 + 1))
                higher_trend_ok = pd.Series(higher_trend_ok, index=df.index)
            else:
                higher_trend_ok = pd.Series(True, index=df.index)

        # --- Señales base EMA crossover ---
        signals = pd.Series(0, index=df.index)

        buy_condition = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['ema_fast'].shift() <= df['ema_slow'].shift()) &
            (df['ema_diff_pct'] > self.signal_threshold) &
            higher_trend_ok
        )
        signals[buy_condition] = 1

        sell_condition = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['ema_fast'].shift() >= df['ema_slow'].shift()) &
            (df['ema_diff_pct'] < -self.signal_threshold) &
            higher_trend_ok
        )
        signals[sell_condition] = -1

        # Mantener posición (relleno hacia adelante)
        if self.keep_position:
            signals = signals.replace(0, np.nan).ffill().fillna(0)
        else:
            signals = signals.fillna(0)
        signals[(df['ema_fast'].isna()) | (df['ema_slow'].isna())] = 0

        # Si quieres solo señales en el momento del cruce:
        # signals = signals.fillna(0)
        # Si quieres mantener posición, deja el ffill pero advierte que puede sobreoperar en mercados laterales.

        # --- Metadatos ---
        current_fast = df['ema_fast'].iloc[-1] if not df['ema_fast'].empty else 0
        current_slow = df['ema_slow'].iloc[-1] if not df['ema_slow'].empty else 0
        current_diff_pct = df['ema_diff_pct'].iloc[-1] if not df['ema_diff_pct'].empty else 0
        last_signal = signals.iloc[-1] if not signals.empty else 0
        signal_changes = (signals != signals.shift()).sum()

        metadata = {
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'signal_threshold': self.signal_threshold,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'current_fast_ema': round(current_fast, 6),
            'current_slow_ema': round(current_slow, 6),
            'current_diff_pct': round(current_diff_pct * 100, 4),
            'last_signal': int(last_signal),
            'signal_changes': int(signal_changes),
            'trend': 'bullish' if current_fast > current_slow else 'bearish',
            'context_timeframes': list(context_data.keys()) if context_data else ['H1']
        }

        signal_data = SignalData(
            signals=signals,
            metadata=metadata,
            strategy_name='ema',
            timestamp=pd.Timestamp.now()
        )

        self.logger.debug(f"[DEBUG] Señales generadas: {signals.value_counts().to_dict()}")
        self.logger.debug(f"[DEBUG] Parámetros: ema_fast={self.ema_fast}, ema_slow={self.ema_slow}, threshold={self.signal_threshold}, stop_loss={self.stop_loss}, take_profit={self.take_profit}")
        self.logger.debug(f"[DEBUG] higher_trend_ok: {higher_trend_ok.value_counts().to_dict()}")
        self.logger.debug(f"[DEBUG] Metadata: {metadata}")
        return self.postprocess_signals(signal_data)

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for optimization"""
        return {
            'ema_fast': 12,
            'ema_slow': 26,
            'signal_threshold': 0.001
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate EMA-specific parameters"""
        super().validate_parameters(params)
        
        ema_fast = params.get('ema_fast', 12)
        ema_slow = params.get('ema_slow', 26)
        
        if ema_fast >= ema_slow:
            raise StrategyError("Fast EMA period must be less than slow EMA period")
    
    def get_required_periods(self) -> int:
        """Get minimum required periods"""
        return max(self.ema_fast, self.ema_slow) + 10  # Extra buffer for EMA stabilization
    
    def get_param_space(self) -> Dict[str, List]:
        """Get parameter space for optimization"""
        return {
            'ema_fast': list(range(5, 20)),  # 5-19
            'ema_slow': list(range(20, 50)), # 20-49
            'signal_threshold': [0.0005, 0.001, 0.002, 0.005, 0.01]
        }
    
    def extract_features(self, data: TradingData) -> pd.DataFrame:
        """Extract EMA-specific features for ML"""
        # Get base features
        features = super().extract_features(data)
        
        # Add EMA-specific features
        df = self.calculate_indicators(data)
        
        # EMA features
        features['ema_fast'] = df['ema_fast']
        features['ema_slow'] = df['ema_slow']
        features['ema_diff'] = df['ema_diff']
        features['ema_diff_pct'] = df['ema_diff_pct']
        features['ema_slope_fast'] = df['ema_fast'].diff()
        features['ema_slope_slow'] = df['ema_slow'].diff()
        
        return features.dropna()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get EMA strategy information"""
        info = super().get_strategy_info()
        info.update({
            'name': 'EMA',
            'type': 'trend_following',
            'description': 'Exponential Moving Average crossover strategy',
            'signals': {
                'buy': 'Fast EMA crosses above slow EMA',
                'sell': 'Fast EMA crosses below slow EMA',
                'threshold': f'Minimum difference: {self.signal_threshold * 100:.1f}%'
            }
        })
        return info