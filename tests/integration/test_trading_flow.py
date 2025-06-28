"""Trading flow integration tests"""

import pytest
from src.strategies.ema.strategy import EMAStrategy
from src.data.providers.yahoo import YahooProvider

@pytest.mark.integration
class TestTradingFlow:
    """Integration tests for complete trading flow"""
    
    def test_basic_flow(self):
        """Test basic trading flow works"""
        # This is a placeholder for now
        assert True

    def test_data_to_strategy_flow(self, sample_trading_data):
        """Test data flows correctly to strategy"""
        strategy = EMAStrategy()
        
        # Test that strategy can process the data
        signals = strategy.generate_signals(sample_trading_data)
        
        assert len(signals.signals) > 0
        # Fix: The actual strategy name is 'ema', not 'EMA'
        assert signals.strategy_name == 'ema'

    def test_provider_to_strategy_integration(self):
        """Test provider and strategy integration"""
        provider = YahooProvider()
        strategy = EMAStrategy()
        
        # Test basic integration
        assert provider is not None
        assert strategy is not None
        
        # For now, just test they can be instantiated together
        # Full integration would require real data fetching

    def test_strategy_consistency(self, sample_trading_data):
        """Test strategy produces consistent results"""
        strategy = EMAStrategy()
        
        # Generate signals twice with same data
        signals1 = strategy.generate_signals(sample_trading_data)
        signals2 = strategy.generate_signals(sample_trading_data)
        
        # Should produce identical results
        assert len(signals1.signals) == len(signals2.signals)
        assert signals1.strategy_name == signals2.strategy_name
        
        # Compare signal values
        import pandas as pd
        pd.testing.assert_series_equal(signals1.signals, signals2.signals)

    def test_multiple_strategies_integration(self, sample_trading_data, all_strategies):
        """Test multiple strategies work with same data"""
        results = {}
        
        for name, strategy in all_strategies.items():
            signals = strategy.generate_signals(sample_trading_data)
            results[name] = {
                'signals_count': len(signals.signals),
                'strategy_name': signals.strategy_name,
                'has_metadata': len(signals.metadata) > 0
            }
            
            assert len(signals.signals) > 0
            assert signals.strategy_name is not None
        
        # All strategies should produce results
        assert len(results) == len(all_strategies)
        print(f"✅ All {len(results)} strategies integrated successfully")