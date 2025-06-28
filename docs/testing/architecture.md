# 🏗️ Arquitectura del Sistema de Testing

## 📁 Estructura de Directorios

```
tests/
├── __init__.py
├── pytest.ini                 # Configuración de pytest
├── conftest.py                # Fixtures compartidos
├── reports/                   # Reportes de coverage
│   ├── coverage_final/        # HTML reports
│   └── junit/                 # XML reports para CI/CD
├── unit/                      # Tests unitarios
│   ├── test_core_comprehensive.py    # Tests principales de core
│   └── test_core_additional.py       # Tests adicionales de cobertura
└── integration/               # Tests de integración
    └── test_system_integration.py    # Tests end-to-end
```

## 🎯 Tipos de Tests

### 🔧 Unit Tests
**Propósito:** Validar componentes individuales en aislamiento

**Ubicación:** `tests/unit/`

**Características:**
- ✅ Rápidos (milisegundos)
- ✅ Aislados (sin dependencias externas)
- ✅ Específicos (un componente por test)
- ✅ Determinísticos (mismo resultado siempre)

**Cobertura:**
- Core types (`TradingData`, `SignalData`)
- Strategies individuales
- Services en aislamiento
- Error handling

### 🔗 Integration Tests
**Propósito:** Validar interacción entre componentes

**Ubicación:** `tests/integration/`

**Características:**
- ✅ Más lentos (segundos)
- ✅ End-to-end workflows
- ✅ Múltiples componentes
- ✅ Datos reales o realistas

**Cobertura:**
- System workflows completos
- Provider integrations
- Strategy combinations
- Performance under load
- Error propagation

## 📊 Organización por Funcionalidad

### test_core_comprehensive.py
```python
TestCoreTypesComprehensive     # TradingData, SignalData
TestStrategyComprehensive      # Strategy behavior
TestDataIntegrity             # Data validation
TestErrorHandling            # Core error scenarios
TestMemoryAndPerformance     # Performance baselines
```

### test_core_additional.py
```python
TestCoreComponentsCoverage    # Additional components
TestErrorScenarios           # Edge cases
TestServiceIntegration       # Service layer
TestDataPipelineComprehensive # Data pipeline
TestStrategyManager          # Strategy management
TestPerformanceAndMemory     # Performance testing
TestIntegrationWorkflows     # Mini integration tests
```

### test_system_integration.py
```python
TestSystemIntegration        # End-to-end system tests
TestDataProviderIntegration  # Provider integrations
TestStrategyIntegration      # Strategy combinations
TestSystemPerformance       # Large dataset performance
TestErrorHandling           # System-wide error handling
```

## 🧪 Patrones de Testing

### 1. Arrange-Act-Assert (AAA)
```python
def test_strategy_signals(self):
    # Arrange
    data = create_test_data()
    strategy = EMAStrategy()
    
    # Act
    signals = strategy.generate_signals(data)
    
    # Assert
    assert len(signals.signals) > 0
    assert signals.strategy_name == 'EMA'
```

### 2. Fixtures para Data Setup
```python
@pytest.fixture
def sample_trading_data(self):
    """Reusable test data"""
    return TradingData(
        symbol='TEST',
        data=create_sample_dataframe(),
        provider='test'
    )
```

### 3. Parameterized Tests
```python
@pytest.mark.parametrize("strategy_class", [
    EMAStrategy, RSIStrategy, MACDStrategy
])
def test_all_strategies(self, strategy_class):
    strategy = strategy_class()
    # Test common behavior
```

### 4. Error Testing
```python
def test_invalid_input_handling(self):
    strategy = EMAStrategy()
    
    with pytest.raises(ValueError):
        strategy.generate_signals(invalid_data)
```

### 5. Performance Testing
```python
def test_large_dataset_performance(self):
    large_data = create_large_dataset(10000)
    
    start_time = time.time()
    strategy.generate_signals(large_data)
    execution_time = time.time() - start_time
    
    assert execution_time < 5.0  # Performance threshold
```

## 🔄 Test Lifecycle

### 1. Setup Phase
- Crear datos de test
- Inicializar componentes
- Configurar mocks si necesario

### 2. Execution Phase
- Ejecutar función bajo test
- Capturar resultados y excepciones

### 3. Validation Phase
- Verificar resultados esperados
- Validar estado del sistema
- Cleanup si necesario

### 4. Teardown Phase
- Limpiar recursos
- Reset global state
- Garbage collection para performance tests

## 🎯 Estrategias de Testing

### Mock Strategy
```python
# Usar cuando hay dependencias externas
with patch('src.data.providers.yahoo.yfinance') as mock_yf:
    mock_yf.download.return_value = mock_data
    provider = YahooProvider()
    data = provider.get_data('EURUSD')
```

### Real Data Strategy
```python
# Usar para integration tests
provider = YahooProvider()
try:
    data = provider.get_data('EURUSD')
    assert data is not None
except Exception as e:
    pytest.skip(f"Provider unavailable: {e}")
```

### Synthetic Data Strategy
```python
# Usar para casos determinísticos
dates = pd.date_range('2024-01-01', periods=100)
data = pd.DataFrame({
    'close': 100 + np.cumsum(np.random.normal(0, 1, 100))
}, index=dates)
```

## 📈 Coverage Strategy

### Target Coverage por Módulo
- **Core types**: 80%+ (críticos)
- **Strategies**: 85%+ (business logic)
- **Services**: 60%+ (integration focused)
- **Data providers**: 50%+ (external dependencies)
- **Interfaces**: 30%+ (UI components)

### Coverage Exclusions
```python
# En .coveragerc
[run]
omit = 
    */tests/*
    */venv/*
    */migrations/*
    */settings/*
```

## 🔧 Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow
    unit: marks tests as unit tests
```

### conftest.py
```python
import pytest
import pandas as pd
from src.core.types import TradingData

@pytest.fixture
def sample_data():
    """Standard test data fixture"""
    # Implementation
    
@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    # Implementation
```

## 🎯 Best Practices Aplicadas

1. **✅ Tests Independientes**: Cada test puede ejecutarse solo
2. **✅ Nombres Descriptivos**: `test_strategy_generates_signals_with_valid_data`
3. **✅ Fast Feedback**: Unit tests < 100ms, Integration < 5s
4. **✅ Determinísticos**: Mismo input = mismo output
5. **✅ Readable**: Test como documentación del comportamiento
6. **✅ Maintainable**: Fácil de actualizar cuando cambia el código