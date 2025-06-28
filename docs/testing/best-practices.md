# 🎯 Mejores Prácticas de Testing

## 📋 Principios Fundamentales

### 1. F.I.R.S.T. Principles
- **🚀 Fast**: Tests deben ser rápidos
- **🔄 Independent**: Tests independientes entre sí
- **🔁 Repeatable**: Mismos resultados siempre
- **✅ Self-Validating**: Pass/Fail claro
- **⏰ Timely**: Escribir tests oportunamente

### 2. Arrange-Act-Assert (AAA)
```python
def test_ema_strategy_signals(self):
    # Arrange - Preparar datos y objetos
    data = create_trending_data()
    strategy = EMAStrategy({'ema_fast': 12, 'ema_slow': 26})
    
    # Act - Ejecutar acción bajo test
    signals = strategy.generate_signals(data)
    
    # Assert - Verificar resultados
    assert len(signals.signals) == len(data.data)
    assert signals.strategy_name == 'EMA'
    assert 'ema_fast' in signals.metadata
```

### 3. Given-When-Then (BDD Style)
```python
def test_strategy_with_insufficient_data(self):
    # Given insufficient data
    minimal_data = create_data_with_rows(5)
    strategy = EMAStrategy()
    
    # When generating signals
    with pytest.raises(InsufficientDataError):
        # Then should raise appropriate error
        strategy.generate_signals(minimal_data)
```

## 🎯 Naming Conventions

### Test File Naming
```
✅ test_core_comprehensive.py
✅ test_system_integration.py
✅ test_strategy_ema.py

❌ comprehensive_test.py
❌ test_file.py
❌ core_tests.py
```

### Test Function Naming
```python
# ✅ Descriptivo y específico
def test_ema_strategy_generates_buy_signal_on_uptrend(self):
def test_yahoo_provider_handles_invalid_symbol_gracefully(self):
def test_orchestrator_returns_system_status_with_timestamp(self):

# ❌ Vago o genérico
def test_strategy(self):
def test_provider(self):
def test_functionality(self):
```

### Test Class Naming
```python
# ✅ Agrupa tests relacionados
class TestEMAStrategySignalGeneration:
class TestYahooProviderDataRetrieval:
class TestOrchestratorSystemStatus:

# ❌ Demasiado genérico
class TestStrategy:
class TestProvider:
class Tests:
```

## 🧪 Patrones de Testing

### 1. Builder Pattern para Test Data
```python
class TradingDataBuilder:
    def __init__(self):
        self.symbol = 'TEST'
        self.timeframe = 'H1'
        self.rows = 100
        self.trend = 'sideways'
    
    def with_symbol(self, symbol):
        self.symbol = symbol
        return self
    
    def with_uptrend(self):
        self.trend = 'up'
        return self
    
    def with_rows(self, rows):
        self.rows = rows
        return self
    
    def build(self):
        return create_trading_data(
            symbol=self.symbol,
            rows=self.rows,
            trend=self.trend
        )

# Uso en tests
def test_strategy_with_uptrend(self):
    data = TradingDataBuilder().with_uptrend().with_rows(200).build()
    # Test implementation
```

### 2. Factory Pattern para Strategies
```python
class StrategyFactory:
    @staticmethod
    def create_ema_strategy(fast=12, slow=26):
        return EMAStrategy({'ema_fast': fast, 'ema_slow': slow})
    
    @staticmethod
    def create_conservative_rsi():
        return RSIStrategy({'period': 21, 'overbought': 75, 'oversold': 25})

# Uso en tests
def test_conservative_rsi_strategy(self):
    strategy = StrategyFactory.create_conservative_rsi()
    # Test implementation
```

### 3. Fixture Parameterization
```python
@pytest.fixture(params=[
    ('EURUSD', 'H1', 100),
    ('GBPUSD', 'H4', 200),
    ('USDJPY', 'D1', 50)
])
def trading_data_variants(request):
    symbol, timeframe, rows = request.param
    return create_trading_data(symbol, timeframe, rows)

def test_strategy_with_different_data(self, trading_data_variants):
    strategy = EMAStrategy()
    signals = strategy.generate_signals(trading_data_variants)
    assert signals is not None
```

## 🔧 Error Testing Best Practices

### 1. Explicit Error Testing
```python
def test_strategy_with_empty_data_raises_error(self):
    empty_data = TradingData(
        symbol='TEST',
        data=pd.DataFrame(),  # Empty
        provider='test'
    )
    strategy = EMAStrategy()
    
    with pytest.raises(InsufficientDataError) as exc_info:
        strategy.generate_signals(empty_data)
    
    assert "insufficient data" in str(exc_info.value).lower()
```

### 2. Graceful Degradation Testing
```python
def test_provider_handles_network_error_gracefully(self):
    provider = YahooProvider()
    
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.ConnectionError()
        
        result = provider.get_data('EURUSD')
        
        # Should return None or empty data, not crash
        assert result is None or len(result.data) == 0
```

### 3. Boundary Testing
```python
@pytest.mark.parametrize("data_size", [0, 1, 2, 10, 25, 26, 27])
def test_ema_strategy_with_various_data_sizes(self, data_size):
    """Test EMA strategy with different data sizes around minimum requirement"""
    data = create_trading_data_with_size(data_size)
    strategy = EMAStrategy({'ema_slow': 26})  # Requires 26 periods
    
    if data_size < 26:
        with pytest.raises(InsufficientDataError):
            strategy.generate_signals(data)
    else:
        signals = strategy.generate_signals(data)
        assert len(signals.signals) == data_size
```

## 📊 Performance Testing

### 1. Timing Assertions
```python
def test_large_dataset_processing_performance(self):
    large_data = create_trading_data_with_size(10000)
    strategy = EMAStrategy()
    
    start_time = time.time()
    signals = strategy.generate_signals(large_data)
    execution_time = time.time() - start_time
    
    # Performance requirements
    assert execution_time < 5.0  # Must complete in under 5 seconds
    assert len(signals.signals) == 10000
    
    print(f"⚡ Processed {len(large_data.data)} rows in {execution_time:.2f}s")
```

### 2. Memory Testing
```python
def test_memory_usage_stability(self):
    """Ensure memory doesn't leak during processing"""
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple datasets
        for i in range(5):
            data = create_trading_data_with_size(1000)
            strategy = EMAStrategy()
            signals = strategy.generate_signals(data)
            
            # Force cleanup
            del data, strategy, signals
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 10  # Less than 10MB increase
        
    except ImportError:
        pytest.skip("psutil not available")
```

## 🎯 Data Testing Strategies

### 1. Realistic Data Generation
```python
def create_realistic_forex_data(symbol='EURUSD', days=30):
    """Generate realistic forex data with proper characteristics"""
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='h')
    
    # Realistic forex price movement
    base_price = 1.1000 if 'EUR' in symbol else 100.0
    
    # Generate prices with realistic volatility
    returns = np.random.normal(0, 0.001, len(dates))  # 0.1% hourly volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    return TradingData(symbol=symbol, data=data, provider='test')
```

### 2. Edge Case Data
```python
def create_edge_case_data():
    """Create data with edge cases for robust testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    
    # Flat line (no volatility)
    flat_data = pd.DataFrame({
        'open': [100.0] * 100,
        'high': [100.0] * 100,
        'low': [100.0] * 100,
        'close': [100.0] * 100,
        'volume': [1000] * 100
    }, index=dates)
    
    return TradingData(symbol='FLAT', data=flat_data, provider='test')

def create_high_volatility_data():
    """Create highly volatile data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    
    # High volatility data
    prices = 100 + np.cumsum(np.random.normal(0, 5, 100))  # 5% volatility
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.1,  # 10% higher
        'low': prices * 0.9,   # 10% lower
        'close': prices,
        'volume': np.random.randint(10000, 100000, 100)
    }, index=dates)
    
    return TradingData(symbol='VOLATILE', data=data, provider='test')
```

## 🔄 Mocking Best Practices

### 1. Mock External Dependencies
```python
def test_yahoo_provider_with_mock(self):
    """Test provider without external API calls"""
    with patch('yfinance.download') as mock_download:
        # Setup mock response
        mock_data = pd.DataFrame({
            'Open': [1.1, 1.2, 1.3],
            'High': [1.15, 1.25, 1.35],
            'Low': [1.05, 1.15, 1.25],
            'Close': [1.12, 1.22, 1.32],
            'Volume': [1000, 2000, 3000]
        })
        mock_download.return_value = mock_data
        
        provider = YahooProvider()
        data = provider.get_data('EURUSD')
        
        assert data is not None
        assert len(data.data) == 3
        mock_download.assert_called_once()
```

### 2. Mock Time-Dependent Code
```python
def test_trading_data_timestamp(self):
    """Test timestamp is set correctly"""
    fixed_time = datetime(2024, 1, 1, 12, 0, 0)
    
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        
        data = TradingData(
            symbol='TEST',
            data=pd.DataFrame({'close': [1, 2, 3]}),
            provider='test'
        )
        
        assert data.timestamp == fixed_time
```

## 📈 Continuous Improvement

### 1. Test Metrics Tracking
```python
# Track test execution time
pytest_plugins = ['pytest-benchmark']

def test_strategy_performance(benchmark):
    data = create_trading_data()
    strategy = EMAStrategy()
    
    result = benchmark(strategy.generate_signals, data)
    assert result is not None
```

### 2. Coverage Goals
```python
# Set progressive coverage targets
# Current: 35%
# Q1 Goal: 40%
# Q2 Goal: 50%
# Q3 Goal: 60%
```

### 3. Test Review Checklist
```markdown
## Test Review Checklist

- [ ] Test name is descriptive and specific
- [ ] Follows AAA pattern
- [ ] Has single responsibility
- [ ] Is independent of other tests
- [ ] Has appropriate assertions
- [ ] Handles edge cases
- [ ] Performance considerations addressed
- [ ] Error scenarios tested
- [ ] Documentation/comments where needed
- [ ] No hardcoded values (use constants/fixtures)
```

## 🎯 Anti-Patterns to Avoid

### ❌ Don't Do This
```python
# Test too many things
def test_everything(self):
    # Tests 10 different scenarios
    
# Unclear test names
def test_func1(self):
def test_case2(self):

# Tests with side effects
def test_that_modifies_global_state(self):
    global_var = "changed"  # ❌

# Flaky tests
def test_with_random_sleep(self):
    time.sleep(random.randint(1, 5))  # ❌

# Tests that depend on external services without mocking
def test_real_api_call(self):
    response = requests.get('https://api.example.com')  # ❌
```

### ✅ Do This Instead
```python
# Single responsibility
def test_ema_strategy_generates_buy_signal_on_golden_cross(self):
    # Test one specific scenario

# Clear naming
def test_yahoo_provider_returns_none_for_invalid_symbol(self):
    
# Isolated tests
def test_strategy_with_clean_state(self):
    strategy = EMAStrategy()  # Fresh instance
    
# Deterministic tests
def test_with_fixed_data(self):
    data = create_deterministic_data()  # ✅

# Mocked external dependencies
@patch('requests.get')
def test_with_mocked_api(self, mock_get):
    mock_get.return_value = Mock(status_code=200)  # ✅
```