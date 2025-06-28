"""Global test configuration and fixtures"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from src.core.types import TradingData, SignalData
from src.strategies.ema.strategy import EMAStrategy
from src.strategies.rsi.strategy import RSIStrategy
from src.strategies.macd.strategy import MACDStrategy

# Test data fixtures
@pytest.fixture
def sample_trading_data():
    """Create sample trading data for testing"""
    dates = pd.date_range('2024-01-01', periods=500, freq='h')
    
    # Create realistic price data with trend and volatility
    base_price = 1.1000
    returns = np.random.normal(0, 0.01, 500)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, 500)),
        'low': prices * (1 - np.random.uniform(0, 0.01, 500)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    return TradingData(
        symbol='EURUSD',
        timeframe='H1',
        data=data,
        provider='test',
        timestamp=datetime.now()
    )

@pytest.fixture
def sample_signals():
    """Create sample signal data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    signals = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)
    
    return SignalData(
        signals=signals,
        metadata={
            'strategy': 'test',
            'parameters': {'test_param': 1.0}
        },
        strategy_name='test_strategy',
        timestamp=datetime.now()
    )

@pytest.fixture
def ema_strategy():
    """Create EMA strategy instance"""
    return EMAStrategy()

@pytest.fixture
def custom_ema_strategy():
    """Create custom EMA strategy instance"""
    return EMAStrategy({
        'ema_fast': 8,
        'ema_slow': 21,
        'signal_threshold': 0.002
    })

@pytest.fixture
def rsi_strategy():
    """Create RSI strategy instance"""
    return RSIStrategy()

@pytest.fixture
def macd_strategy():
    """Create MACD strategy instance"""
    return MACDStrategy()

@pytest.fixture
def all_strategies():
    """Create all strategy instances"""
    return {
        'ema': EMAStrategy(),
        'rsi': RSIStrategy(),
        'macd': MACDStrategy()
    }

@pytest.fixture
def mock_data_provider():
    """Create mock data provider"""
    mock = Mock()
    mock.get_data.return_value = sample_trading_data()
    mock.is_connected.return_value = True
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    return mock

@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory"""
    data_dir = Path(__file__).parent / "fixtures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@pytest.fixture(scope="session")
def temp_cache_dir(tmp_path_factory):
    """Temporary cache directory for tests"""
    return tmp_path_factory.mktemp("test_cache")

# Performance fixtures
@pytest.fixture
def benchmark_data():
    """Create benchmark data for performance tests"""
    dates = pd.date_range('2020-01-01', periods=10000, freq='h')
    prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, 10000))
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.randint(1000, 5000, 10000)
    }, index=dates)
    
    return TradingData(
        symbol='EURUSD',
        timeframe='H1',
        data=data,
        provider='test',
        timestamp=datetime.now()
    )

# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker by default
        if not any(marker.name in ['integration', 'performance'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to performance tests
        if any(marker.name == 'performance' for marker in item.iter_markers()):
            item.add_marker(pytest.mark.slow)