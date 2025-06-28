"""Simple core functionality tests"""

import pytest
import pandas as pd
from datetime import datetime

from src.core.types import TradingData, SignalData
from src.core.exceptions import StrategyError, DataProviderError

@pytest.mark.unit
class TestCoreTypes:
    """Test core data types"""
    
    def test_trading_data_creation(self):
        """Test TradingData creation"""
        data = pd.DataFrame({
            'open': [1.1, 1.2],
            'high': [1.15, 1.25],
            'low': [1.05, 1.15],
            'close': [1.12, 1.22],
            'volume': [1000, 2000]
        })
        
        trading_data = TradingData(
            symbol='TEST',
            timeframe='H1',
            data=data,
            provider='test',
            timestamp=datetime.now()
        )
        
        assert trading_data.symbol == 'TEST'
        assert trading_data.timeframe == 'H1'
        assert len(trading_data.data) == 2

    def test_signal_data_creation(self):
        """Test SignalData creation"""
        signals = pd.Series([1, -1, 0, 1])
        
        signal_data = SignalData(
            signals=signals,
            metadata={'test': 'value'},
            strategy_name='test_strategy',
            timestamp=datetime.now()
        )
        
        assert len(signal_data.signals) == 4
        assert signal_data.strategy_name == 'test_strategy'
        assert 'test' in signal_data.metadata

    def test_trading_data_with_sample_fixture(self, sample_trading_data):
        """Test with sample fixture"""
        assert isinstance(sample_trading_data, TradingData)
        assert sample_trading_data.symbol == 'EURUSD'
        assert len(sample_trading_data.data) > 0

@pytest.mark.unit
class TestCoreExceptions:
    """Test core exceptions"""
    
    def test_strategy_error(self):
        """Test StrategyError can be raised"""
        with pytest.raises(StrategyError):
            raise StrategyError("Test error")

    def test_data_provider_error(self):
        """Test DataProviderError can be raised"""
        with pytest.raises(DataProviderError):
            raise DataProviderError("Test error")

    def test_exception_messages(self):
        """Test exception messages are preserved"""
        message = "Custom error message"
        
        try:
            raise StrategyError(message)
        except StrategyError as e:
            assert str(e) == message
        
        try:
            raise DataProviderError(message)
        except DataProviderError as e:
            assert str(e) == message