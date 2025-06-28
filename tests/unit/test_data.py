"""Data provider tests"""

import pytest
from src.data.providers.yahoo import YahooProvider

class TestDataProviders:
    def test_yahoo_provider_initialization(self):
        """Test Yahoo provider initialization"""
        provider = YahooProvider()
        assert provider is not None

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
        assert len(symbols) > 0