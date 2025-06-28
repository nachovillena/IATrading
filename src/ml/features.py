"""Feature engineering for trading data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ..core.types import TradingData
from ..core.exceptions import TradingSystemError
from ..utils.logger import Logger

class FeatureEngineer:
    """Creates technical features from trading data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer"""
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
    
    def create_features(self, data: TradingData) -> pd.DataFrame:
        """Create technical features from trading data"""
        try:
            df = data.data.copy()
            self.logger.info(f"Creating features for {len(df)} records")
            
            # Price-based features
            df = self._add_price_features(df)
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Time-based features
            df = self._add_time_features(df)
            
            # Volatility features
            df = self._add_volatility_features(df)
            
            # Remove NaN rows (from rolling calculations)
            df_clean = df.dropna()
            
            feature_count = len([col for col in df_clean.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
            self.logger.info(f"Created {feature_count} features for {len(df_clean)} records")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Feature creation failed: {e}")
            raise TradingSystemError(f"Feature engineering failed: {e}")
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # OHLC relationships
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Session indicators (assuming UTC time)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['new_york_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # Historical volatility
        for period in [10, 20]:
            df[f'volatility_{period}'] = df['price_change'].rolling(window=period).std()
        
        # Average True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift()).abs()
        df['tr3'] = (df['low'] - df['close'].shift()).abs()
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Clean up temporary columns
        df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
