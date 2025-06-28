"""Simple data provider tests"""

import pytest
from src.data.providers.yahoo import YahooProvider
from src.core.exceptions import DataProviderError

@pytest.mark.unit
class TestDataProviders:
    """Basic data provider tests"""
    
    def test_yahoo_provider_initialization(self):
        """Test Yahoo provider can be initialized"""
        provider = YahooProvider()
        assert provider is not None
        assert hasattr(provider, 'get_data')

    def test_yahoo_provider_basic_methods(self):
        """Test Yahoo provider has required methods"""
        provider = YahooProvider()
        
        # Check basic interface
        assert hasattr(provider, 'connect')
        assert hasattr(provider, 'disconnect')
        assert hasattr(provider, 'is_connected')
        
        # Test connection methods don't crash
        assert provider.connect() in [True, False]
        # Fix: is_connected is a property, not a method
        assert provider.is_connected in [True, False]
        provider.disconnect()

    def test_yahoo_provider_get_available_symbols(self):
        """Test getting available symbols"""
        provider = YahooProvider()
        symbols = provider.get_available_symbols()
        
        assert isinstance(symbols, list)
        # Should have at least some common symbols
        assert len(symbols) > 0

    def test_yahoo_provider_provider_info(self):
        """Test provider info"""
        provider = YahooProvider()
        
        # Test basic attributes
        assert hasattr(provider, 'provider_id')
        assert provider.provider_id == 'yahoo'
        
        # Test string representation
        provider_str = str(provider)
        assert 'yahoo' in provider_str.lower()