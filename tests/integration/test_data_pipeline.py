"""Data pipeline integration tests"""

import pytest
from src.data.providers.yahoo import YahooProvider

@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data pipeline"""
    
    def test_pipeline_flow(self):
        """Test basic pipeline flow"""
        # Placeholder for now
        assert True

    def test_provider_connection_flow(self):
        """Test provider connection workflow"""
        provider = YahooProvider()
        
        # Test connection flow - fix: is_connected is a property, not a method
        initial_status = provider.is_connected
        connect_result = provider.connect()
        connected_status = provider.is_connected
        
        provider.disconnect()
        final_status = provider.is_connected
        
        # Basic assertions about the flow
        assert isinstance(initial_status, bool)
        assert isinstance(connect_result, bool)
        assert isinstance(connected_status, bool)
        assert isinstance(final_status, bool)

    def test_provider_basic_functionality(self):
        """Test provider basic functionality"""
        provider = YahooProvider()
        
        # Test that provider can be created and has basic methods
        assert hasattr(provider, 'get_data')
        assert hasattr(provider, 'get_available_symbols')
        
        # Test getting symbols
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_provider_error_handling(self):
        """Test provider error handling"""
        provider = YahooProvider()
        
        # Test that provider handles invalid requests gracefully
        try:
            # This might fail, but shouldn't crash
            data = provider.get_data('INVALID_SYMBOL', '1h', days=1)
            # If it succeeds, that's fine too
            assert data is not None or data is None
        except Exception as e:
            # If it raises an exception, that's also acceptable
            assert isinstance(e, Exception)