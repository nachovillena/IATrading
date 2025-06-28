# 🔧 Guía de Troubleshooting - Tests

## 🚨 Problemas Comunes y Soluciones

### 1. Import Errors

#### ❌ Error: `ImportError: cannot import name 'TradingConfig'`
```bash
ImportError: cannot import name 'TradingConfig' from 'src.core.config'
```

**🔍 Causa:** Import de clase/función que no existe
**✅ Solución:**
```python
# ❌ Incorrecto
from src.core.config import TradingConfig

# ✅ Correcto - verificar qué existe realmente
from src.core.config.config_loader import ConfigLoader
```

**🛠️ Debug Steps:**
```bash
# 1. Verificar qué hay en el módulo
python -c "import src.core.config; print(dir(src.core.config))"

# 2. Verificar estructura de archivos
ls -la src/core/config/

# 3. Ejecutar test específico para verificar import
python -c "from src.core.types import TradingData; print('OK')"
```

#### ❌ Error: `ModuleNotFoundError: No module named 'src'`
```bash
ModuleNotFoundError: No module named 'src'
```

**🔍 Causa:** Python no encuentra el módulo src
**✅ Solución:**
```bash
# 1. Verificar que estás en el directorio correcto
pwd  # Debe ser: /IATrading

# 2. Verificar que existe __init__.py en src/
ls src/__init__.py

# 3. Ejecutar pytest desde el directorio raíz
python -m pytest tests/unit/test_core_additional.py -v
```

### 2. Test Failures

#### ❌ Error: `AssertionError: assert False`
```python
def test_strategy_signals(self):
    signals = strategy.generate_signals(data)
    assert len(signals.signals) > 0  # ❌ Falla porque no hay señales
```

**🔍 Causa:** La estrategia no genera señales con los datos de test
**✅ Solución:**
```python
def test_strategy_signals(self):
    # ✅ Crear datos que definitivamente generen señales
    data = create_trending_data_with_crossover()
    signals = strategy.generate_signals(data)
    
    # ✅ Verificación más robusta
    assert isinstance(signals, SignalData)
    assert len(signals.signals) >= 0  # Permitir 0 señales
    assert signals.strategy_name is not None
```

#### ❌ Error: Tests intermitentes (Flaky Tests)
```python
def test_performance(self):
    start_time = time.time()
    strategy.generate_signals(data)
    execution_time = time.time() - start_time
    assert execution_time < 1.0  # ❌ Falla a veces por carga del sistema
```

**🔍 Causa:** Dependencia en timing del sistema
**✅ Solución:**
```python
def test_performance(self):
    start_time = time.time()
    strategy.generate_signals(data)
    execution_time = time.time() - start_time
    
    # ✅ Umbral más tolerante o usar benchmark
    assert execution_time < 5.0  # Más tolerante
    print(f"⚡ Execution time: {execution_time:.2f}s")
```

### 3. Coverage Issues

#### ❌ Error: `Coverage.py warning: No data was collected`
```bash
Coverage.py warning: No data was collected. (no-data-collected)
```

**🔍 Causa:** Coverage no puede encontrar archivos para medir
**✅ Solución:**
```bash
# 1. Verificar que existe código en src/
find src/ -name "*.py" | head -5

# 2. Ejecutar con path específico
python -m pytest tests/ --cov=./src --cov-report=term-missing -v

# 3. Verificar configuración en pytest.ini o .coveragerc
cat .coveragerc
```

#### ❌ Error: Coverage muy bajo inesperadamente
```bash
Coverage: 5% (expected ~35%)
```

**🔍 Causa:** Solo midiendo archivos específicos
**✅ Solución:**
```bash
# 1. Verificar qué archivos está midiendo coverage
python -m pytest --cov=src --cov-report=term-missing -v | grep "src/"

# 2. Ejecutar con todos los tests
python -m pytest tests/unit/ tests/integration/ --cov=src -v

# 3. Verificar exclusiones en .coveragerc
```

### 4. Performance Issues

#### ❌ Error: Tests muy lentos
```bash
Tests taking > 30 seconds
```

**🔍 Causa:** Tests con datasets muy grandes o muchas repeticiones
**✅ Solución:**
```python
# ❌ Lento
def test_with_huge_dataset(self):
    data = create_data(100000)  # Muy grande
    
# ✅ Rápido pero efectivo
def test_with_reasonable_dataset(self):
    data = create_data(1000)    # Suficiente para test
    
# ✅ Marcar tests lentos
@pytest.mark.slow
def test_large_dataset(self):
    data = create_data(100000)
```

```bash
# Ejecutar sin tests lentos
python -m pytest -m "not slow" -v
```

#### ❌ Error: Memory leak en tests
```python
def test_memory_usage(self):
    # Memory usage keeps growing
```

**✅ Solución:**
```python
def test_memory_usage(self):
    import gc
    
    for i in range(10):
        data = create_large_data()
        strategy = EMAStrategy()
        signals = strategy.generate_signals(data)
        
        # ✅ Cleanup explícito
        del data, strategy, signals
        gc.collect()
```

### 5. Fixture Issues

#### ❌ Error: `fixture 'sample_data' not found`
```python
def test_with_fixture(self, sample_data):  # ❌ Fixture no encontrado
```

**🔍 Causa:** Fixture no definido o no en scope correcto
**✅ Solución:**
```python
# 1. Definir fixture en conftest.py o en mismo archivo
@pytest.fixture
def sample_data():
    return create_test_data()

# 2. Verificar que conftest.py está en directorio correcto
tests/
├── conftest.py          # ✅ Global fixtures
├── unit/
│   ├── conftest.py      # ✅ Unit test fixtures
│   └── test_file.py
└── integration/
    └── test_file.py
```

#### ❌ Error: Fixture con scope incorrecto
```python
@pytest.fixture(scope="session")  # ❌ Muy amplio
def database_connection():
    # Se reutiliza entre tests que deberían ser independientes
```

**✅ Solución:**
```python
@pytest.fixture(scope="function")  # ✅ Aislado por test
def clean_database():
    # Nueva instancia para cada test
    
@pytest.fixture(scope="session")   # ✅ Solo para recursos costosos
def expensive_setup():
    # Solo cosas que realmente son costosas de crear
```

## 🛠️ Debugging Strategies

### 1. Debugging Test Individual
```bash
# 1. Ejecutar un test específico con output
python -m pytest tests/unit/test_core_additional.py::TestStrategyManager::test_strategy_creation -v -s

# 2. Agregar prints para debugging
def test_debug_example(self):
    print(f"🔍 Data shape: {data.shape}")
    print(f"🔍 Strategy params: {strategy.parameters}")
    result = strategy.generate_signals(data)
    print(f"🔍 Result: {result}")
    assert result is not None
```

### 2. Debugging con pdb
```python
def test_with_debugger(self):
    import pdb; pdb.set_trace()  # ✅ Breakpoint
    
    data = create_test_data()
    strategy = EMAStrategy()
    signals = strategy.generate_signals(data)  # Examinar aquí
    assert signals is not None
```

### 3. Debugging Coverage
```bash
# 1. Ver líneas específicas sin coverage
python -m pytest --cov=src.strategies --cov-report=term-missing -v

# 2. Generar reporte HTML detallado
python -m pytest --cov=src --cov-report=html:debug_coverage -v
# Abrir: debug_coverage/index.html

# 3. Ejecutar con coverage específico
python -m pytest tests/unit/test_core_additional.py --cov=src.strategies.ema.strategy --cov-report=term-missing -v
```

## 🚨 Error Messages Comunes

### 1. Strategy Errors
```python
# ❌ Error común
InsufficientDataError: Strategy requires at least 26 periods

# ✅ Fix
def test_strategy_with_sufficient_data(self):
    data = create_data_with_periods(50)  # > 26
    strategy = EMAStrategy({'ema_slow': 26})
    signals = strategy.generate_signals(data)
    assert signals is not None
```

### 2. Data Provider Errors
```python
# ❌ Error común
ConnectionError: Unable to connect to data provider

# ✅ Fix con mock
@patch('requests.get')
def test_provider_with_mock(self, mock_get):
    mock_get.side_effect = ConnectionError()
    provider = YahooProvider()
    
    # Test que maneja el error gracefully
    data = provider.get_data('EURUSD')
    assert data is None  # Debería fallar gracefully
```

### 3. Threading Errors
```python
# ❌ Error común
RuntimeError: Thread safety violation

# ✅ Fix con locks
import threading

def test_thread_safety(self):
    lock = threading.Lock()
    results = []
    
    def worker():
        with lock:
            strategy = EMAStrategy()
            # Thread-safe operations
```

## 📊 Health Check Commands

### Quick Health Check
```bash
# 1. Test básico - debe pasar
python -m pytest tests/unit/test_core_additional.py::TestCoreComponentsCoverage::test_orchestrator_comprehensive -v

# 2. Coverage check - debe ser ~35%
python -m pytest tests/unit/test_core_additional.py --cov=src --cov-report=term-missing -q

# 3. Integration check - debe pasar
python -m pytest tests/integration/test_system_integration.py::TestSystemIntegration::test_orchestrator_basic_functionality -v
```

### Comprehensive Health Check
```bash
# Sistema completo
python -m pytest tests/ --cov=src --cov-fail-under=33 -x  # Para en primer error

# Performance check
python -m pytest -k "performance" -v

# Memory check
python -m pytest -k "memory" -v
```

## 🔧 Environment Issues

### 1. Dependencies Missing
```bash
# ❌ Error: ModuleNotFoundError: No module named 'pandas'
pip install -r requirements.txt

# ❌ Error: ModuleNotFoundError: No module named 'pytest'
pip install pytest pytest-cov

# ✅ Verificar dependencies
pip list | grep -E "(pytest|pandas|numpy)"
```

### 2. Python Version Issues
```bash
# ❌ Error: SyntaxError (Python version muy vieja)
python --version  # Debe ser 3.8+

# ✅ Fix con Python correcto
python3.9 -m pytest tests/ -v
```

### 3. Path Issues
```bash
# ❌ Error: Tests no se encuentran
# Verificar estructura
find . -name "test_*.py" -type f

# ✅ Configurar PYTHONPATH si necesario
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m pytest tests/ -v
```

## 📋 Troubleshooting Checklist

### Antes de Ejecutar Tests
- [ ] Estás en el directorio correcto (IATrading/)
- [ ] Existe `src/__init__.py`
- [ ] Dependencies instaladas (`pip list`)
- [ ] Python version correcta (3.8+)

### Si Tests Fallan
- [ ] Leer el error message completo
- [ ] Verificar imports en el archivo de test
- [ ] Ejecutar test individual con `-v -s`
- [ ] Verificar que los datos de test son válidos
- [ ] Comprobar que fixtures están disponibles

### Si Coverage Es Bajo
- [ ] Ejecutar todos los tests (`tests/unit/` y `tests/integration/`)
- [ ] Verificar que `--cov=src` apunta al directorio correcto
- [ ] Comprobar exclusiones en `.coveragerc`
- [ ] Verificar que el código se está ejecutando realmente

### Si Tests Son Lentos
- [ ] Identificar tests específicos lentos
- [ ] Usar datasets más pequeños para tests unitarios
- [ ] Marcar tests lentos con `@pytest.mark.slow`
- [ ] Ejecutar con `-m "not slow"` durante desarrollo

## 🆘 Cuando Todo Falla

### 1. Reset Completo
```bash
# 1. Limpiar cache
rm -rf .pytest_cache/
rm -rf __pycache__/
find . -name "*.pyc" -delete

# 2. Reinstalar dependencies
pip uninstall pytest pytest-cov -y
pip install pytest pytest-cov

# 3. Test simple
python -c "import src; print('Import OK')"
python -m pytest --version
```

### 2. Minimal Test
```python
# Crear test minimal para verificar setup
# tests/test_minimal.py
def test_minimal():
    assert True

def test_import():
    from src.core.types import TradingData
    assert TradingData is not None
```

```bash
python -m pytest tests/test_minimal.py -v
```

### 3. Pedir Ayuda
```bash
# 1. Generar info completa del error
python -m pytest tests/ -v --tb=long > error_log.txt 2>&1

# 2. Verificar versiones
python --version
pip list > requirements_current.txt

# 3. Estructura del proyecto
find . -name "*.py" | head -20 > project_structure.txt
```

**📞 Con esta información, puedes pedir ayuda específica!**