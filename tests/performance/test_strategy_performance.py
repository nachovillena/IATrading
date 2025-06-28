"""Strategy performance tests"""

import pytest
import time

from src.strategies.ema.strategy import EMAStrategy
from src.strategies.rsi.strategy import RSIStrategy
from src.strategies.macd.strategy import MACDStrategy

@pytest.mark.performance
class TestStrategyPerformance:
    """Performance tests for trading strategies"""
    
    def test_ema_performance_small_dataset(self, sample_trading_data):
        """Test EMA performance with small dataset"""
        strategy = EMAStrategy()
        
        start_time = time.perf_counter()
        result = strategy.generate_signals(sample_trading_data)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        assert len(result.signals) > 0
        assert execution_time < 0.1  # Should be very fast
        print(f"EMA (small): {execution_time:.4f}s")

    def test_ema_performance_large_dataset(self, benchmark_data):
        """Test EMA performance with large dataset"""
        strategy = EMAStrategy()
        
        start_time = time.perf_counter()
        result = strategy.generate_signals(benchmark_data)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        assert len(result.signals) > 0
        assert execution_time < 1.0  # Should complete in reasonable time
        print(f"EMA (large): {execution_time:.4f}s")

    def test_all_strategies_performance_comparison(self, sample_trading_data):
        """Compare performance of all strategies"""
        strategies = {
            'EMA': EMAStrategy(),
            'RSI': RSIStrategy(),
            'MACD': MACDStrategy()
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            start_time = time.perf_counter()
            signals = strategy.generate_signals(sample_trading_data)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            results[name] = {
                'time': execution_time,
                'signals_count': len(signals.signals)
            }
            
            assert len(signals.signals) > 0
            print(f"{name}: {execution_time:.4f}s ({len(signals.signals)} signals)")
        
        # All should complete reasonably fast
        for name, result in results.items():
            assert result['time'] < 0.5

    def test_memory_usage_basic(self, benchmark_data):
        """Basic memory usage test"""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        strategy = EMAStrategy()
        result = strategy.generate_signals(benchmark_data)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert len(result.signals) > 0
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")