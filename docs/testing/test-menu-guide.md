# 🧪 Guía Completa del Test Menu Interactivo

## 📋 Índice

1. [Introducción](#introducción)
2. [Instalación y Configuración](#instalación-y-configuración)
3. [Menú Principal](#menú-principal)
4. [Menús Específicos](#menús-específicos)
5. [Funciones Avanzadas](#funciones-avanzadas)
6. [Casos de Uso](#casos-de-uso)
7. [Troubleshooting](#troubleshooting)

## 🎯 Introducción

El **Test Menu Interactivo** (`test_menu.py`) es una herramienta centralizada que proporciona una interfaz amigable para ejecutar todos los tipos de tests en IATrading. Elimina la necesidad de recordar comandos complejos de pytest y organiza todas las opciones de testing en menús intuitivos.

### ✨ Características Principales

- **🎯 Interfaz Interactiva**: Menús organizados y fáciles de navegar
- **📊 Reportes Automáticos**: Genera reportes HTML, XML y de coverage automáticamente
- **🕐 Timestamping**: Cada ejecución crea un directorio con timestamp único
- **🔗 Symlinks Inteligentes**: Siempre mantiene un enlace al reporte más reciente
- **🧹 Limpieza Automática**: Herramientas para mantener el entorno limpio
- **📈 Visualización de Resultados**: Abre reportes directamente en el navegador
- **🔧 Multiplataforma**: Funciona en Windows, macOS y Linux

## 🚀 Instalación y Configuración

### Prerrequisitos

```bash
# Instalar dependencias
pip install pytest pytest-html pytest-cov pytest-benchmark

# Verificar instalación
python -m pytest --version
```

### Estructura de Directorios

El test menu creará automáticamente esta estructura:

```
tests/
├── test_menu.py                 # Archivo principal
├── reports/                     # Directorio de reportes
│   ├── latest/                  # Symlink al reporte más reciente
│   ├── 2024-01-15_10-30-45/    # Reporte timestamped
│   │   ├── htmlcov/             # Coverage HTML
│   │   ├── logs/                # Logs específicos
│   │   ├── pytest_report.html  # Reporte principal
│   │   ├── coverage.xml         # Coverage XML
│   │   └── test_output.txt      # Output completo
│   └── ...                     # Otros reportes
├── unit/                        # Tests unitarios
├── integration/                 # Tests de integración
└── performance/                 # Tests de performance
```

### Ejecución

```bash
# Desde el directorio tests/
python test_menu.py

# O desde cualquier lugar
python tests/test_menu.py
```

## 🎯 Menú Principal

```
🧪 IATRADING TEST SUITE
======================================
📁 Current Report: 2024-01-15_10-30-45
📊 Reports Dir: /path/to/reports/
======================================
1. 🎯 Unit Tests
2. 🔗 Integration Tests
3. ⚡ Performance Tests
4. 📊 Full Test Suite
5. 🧹 Clean Test Environment
6. 📈 View Test Reports
7. 🔧 Test Configuration
8. 📋 Test Status & Info
0. ❌ Exit
======================================
```

### 📖 Descripción de Opciones Principales

#### **1. 🎯 Unit Tests**
**Propósito:** Acceso a todos los tests unitarios organizados por categoría

**Cuándo usar:**
- ✅ Desarrollo diario de código
- ✅ Testing de componentes específicos
- ✅ Validación rápida de cambios
- ✅ Debugging detallado

**Tiempo típico:** 30 segundos - 2 minutos

#### **2. 🔗 Integration Tests**
**Propósito:** Tests que validan la interacción entre componentes

**Cuándo usar:**
- ✅ Verificar workflows completos
- ✅ Testing de flujos de datos
- ✅ Validación antes de releases
- ✅ Testing end-to-end

**Tiempo típico:** 1-5 minutos

#### **3. ⚡ Performance Tests**
**Propósito:** Validación de rendimiento y benchmarks

**Cuándo usar:**
- ✅ Optimización de código
- ✅ Comparación de versiones
- ✅ Análisis de bottlenecks
- ✅ Validación de SLAs

**Tiempo típico:** 2-10 minutos

#### **4. 📊 Full Test Suite**
**Propósito:** Combinaciones completas de tests

**Cuándo usar:**
- ✅ Validación completa pre-release
- ✅ CI/CD pipelines
- ✅ Verificación exhaustiva
- ✅ Reportes de calidad

**Tiempo típico:** 5-20 minutos

#### **5. 🧹 Clean Test Environment**
**Propósito:** Limpieza y mantenimiento del entorno

**Cuándo usar:**
- ✅ Problemas de espacio en disco
- ✅ Comportamiento extraño de tests
- ✅ Mantenimiento semanal
- ✅ Preparar entorno limpio

#### **6. 📈 View Test Reports**
**Propósito:** Visualización de reportes históricos

**Cuándo usar:**
- ✅ Análisis de resultados pasados
- ✅ Comparación de coverage
- ✅ Presentaciones
- ✅ Debugging histórico

#### **7. 🔧 Test Configuration**
**Propósito:** Gestión de configuración de testing

**Cuándo usar:**
- ✅ Configurar nuevos tests
- ✅ Modificar parámetros de pytest
- ✅ Debugging de configuración
- ✅ Setup inicial

#### **8. 📋 Test Status & Info**
**Propósito:** Información general del sistema de testing

**Cuándo usar:**
- ✅ Overview rápido del proyecto
- ✅ Verificar conteo de tests
- ✅ Status check general
- ✅ Información para reportes

## 🎯 Menús Específicos

### 🎯 Unit Tests Menu

```
🎯 UNIT TESTS
--------------------------------------------------
1. 📈 Strategy Tests
   - EMA Strategy
   - RSI Strategy
   - MACD Strategy
2. 💾 Data Provider Tests
3. 🧠 Core Functionality Tests
4. 🖥️  Interface Tests
5. 🎯 All Unit Tests (Quick)
6. 🎯 All Unit Tests (Verbose)
7. 🎯 All Unit Tests (Coverage)
8. 🧪 Custom Test Selection
0. ⬅️  Back to Main Menu
--------------------------------------------------
```

#### **Opciones Detalladas:**

##### **1. 📈 Strategy Tests**
Accede al submenú de estrategias:

```
📈 STRATEGY TESTS
----------------------------------------
1. 📊 EMA Strategy Tests
2. 📈 RSI Strategy Tests
3. 📉 MACD Strategy Tests
4. 🎯 All Strategy Tests
5. ⚡ Strategy Performance Compare
0. ⬅️  Back
----------------------------------------
```

**Comandos ejecutados:**
- **EMA**: `pytest unit/test_strategies.py::TestEMAStrategy -v`
- **RSI**: `pytest unit/test_strategies.py::TestRSIStrategy -v`
- **MACD**: `pytest unit/test_strategies.py::TestMACDStrategy -v`
- **All**: `pytest unit/test_strategies.py -v`
- **Performance**: `pytest unit/test_strategies.py::TestStrategyPerformance -v`

##### **2. 💾 Data Provider Tests**
**Comando:** `pytest unit/test_data.py -v`

**Crea automáticamente** si no existe:
```python
"""Data provider tests"""
import pytest
from src.data.providers.yahoo import YahooProvider

class TestDataProviders:
    def test_yahoo_provider_init(self):
        provider = YahooProvider()
        assert provider is not None
```

##### **3. 🧠 Core Functionality Tests**
**Comando:** `pytest unit/test_core.py -v`

**Crea automáticamente** si no existe:
```python
"""Core functionality tests"""
import pytest
from src.core.types import TradingData

class TestCore:
    def test_trading_data_creation(self):
        assert TradingData is not None
```

##### **4. 🖥️ Interface Tests**
**Comando:** `pytest unit/test_interfaces.py -v`

**Crea automáticamente** si no existe:
```python
"""Interface tests"""
import pytest
from src.interfaces.menu_interface import MenuInterface

class TestInterfaces:
    def test_menu_interface_init(self):
        menu = MenuInterface()
        assert menu is not None
```

##### **5. 🎯 All Unit Tests (Quick)**
**Comando:** `pytest unit/ -q`
**Tiempo:** ~30 segundos
**Uso:** Check rápido sin detalles

##### **6. 🎯 All Unit Tests (Verbose)**
**Comando:** `pytest unit/ -v`
**Tiempo:** ~45 segundos
**Uso:** Ver progreso detallado de cada test

##### **7. 🎯 All Unit Tests (Coverage)**
**Comando:** 
```bash
pytest unit/ -v \
  --cov=src \
  --cov-report=html:reports/TIMESTAMP/htmlcov \
  --cov-report=term \
  --cov-report=xml:reports/TIMESTAMP/coverage.xml
```
**Tiempo:** ~1-2 minutos
**Genera:** Reportes HTML, XML y terminal de coverage

##### **8. 🧪 Custom Test Selection**
**Interactivo:** Te permite escribir comandos pytest personalizados

**Ejemplos de uso:**
```bash
# Test específico
unit/test_strategies.py::TestEMAStrategy::test_initialization_default

# Test con keyword
unit/test_core.py -k "trading_data"

# Stop en primer fallo
unit/ -x --tb=short
```

### 🔗 Integration Tests Menu

```
🔗 Integration Tests
1. 🔄 Trading Flow Tests
2. 📊 Data Pipeline Tests
3. 🔗 All Integration Tests
```

**Comandos ejecutados:**
- **Trading Flow**: `pytest integration/test_trading_flow.py -v`
- **Data Pipeline**: `pytest integration/test_data_pipeline.py -v`
- **All Integration**: `pytest integration/ -v`

**Auto-creación:** Si los archivos no existen, los crea con tests placeholder.

### ⚡ Performance Tests Menu

```
⚡ Performance Tests
1. 📈 Strategy Performance
2. 💾 Data Loading Performance
3. ⚡ All Performance Tests
4. 🎯 Benchmark Comparison
```

**Comandos ejecutados:**
- **Strategy**: `pytest performance/test_strategy_performance.py -v`
- **Data Loading**: `pytest performance/test_data_performance.py -v`
- **All Performance**: `pytest performance/ -v`
- **Benchmarks**: `pytest performance/ --benchmark-compare -v`

### 📊 Full Test Suite Options

```
📊 Full Test Suite Options
1. 🚀 Quick (Unit tests only)
2. 🔄 Standard (Unit + Integration)
3. 🎯 Complete (All tests + Coverage)
4. 📊 Complete + Performance
5. 🧪 Complete + Benchmarks
```

#### **Comandos por Opción:**

##### **1. 🚀 Quick**
```bash
pytest unit/ -v
```
**Tiempo:** ~1 minuto
**Uso:** Validación rápida pre-commit

##### **2. 🔄 Standard**
```bash
pytest unit/ integration/ -v
```
**Tiempo:** ~3 minutos
**Uso:** Validación estándar pre-push

##### **3. 🎯 Complete**
```bash
pytest unit/ integration/ -v \
  --cov=src \
  --cov-report=html:reports/TIMESTAMP/htmlcov \
  --cov-report=term
```
**Tiempo:** ~5 minutos
**Uso:** Validación completa pre-release

##### **4. 📊 Complete + Performance**
```bash
pytest unit/ integration/ performance/ -v \
  --cov=src \
  --cov-report=html:reports/TIMESTAMP/htmlcov
```
**Tiempo:** ~10 minutos
**Uso:** Validación exhaustiva

##### **5. 🧪 Complete + Benchmarks**
```bash
pytest unit/ integration/ performance/ -v \
  --cov=src \
  --cov-report=html:reports/TIMESTAMP/htmlcov \
  --benchmark-only
```
**Tiempo:** ~15 minutos
**Uso:** Análisis de performance completo

## 🧹 Funciones de Limpieza

### Clean Test Environment Options

```
🧹 Cleaning Test Environment...
What would you like to clean?
1. 🗑️  Cache files (__pycache__, .pytest_cache)
2. 📊 Old reports (keep last 5)
3. 🧹 Temporary files (*.tmp, *.log)
4. 🔄 Reset test database/cache
5. 🧹 Clean all
6. 🗂️  Organize old reports
```

#### **Detalles por Opción:**

##### **1. 🗑️ Cache Files**
**Limpia:**
- `**/__pycache__` (bytecode Python)
- `**/.pytest_cache` (cache de pytest)
- `**/data/cache` (cache de datos)
- `**/*.pyc` (archivos compilados)

**Comando equivalente:**
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

##### **2. 📊 Old Reports**
**Funcionalidad:** Mantiene solo los últimos 5 reportes
**Ahorra:** Espacio en disco significativo
**Seguro:** Nunca borra el reporte actual

##### **3. 🧹 Temporary Files**
**Limpia:**
- `**/*.tmp` (archivos temporales)
- `**/*.log~` (logs backup)
- `**/*~` (archivos de backup)

##### **4. 🔄 Reset Test Data**
**Limpia:**
- `tests/fixtures/cache` (cache de fixtures)
- Datos de test cacheados
- Bases de datos de test

##### **5. 🧹 Clean All**
**Ejecuta:** Todas las opciones anteriores en secuencia

##### **6. 🗂️ Organize Reports**
**Funcionalidad:** Organiza reportes por fecha en subdirectorios

## 📈 Visualización de Reportes

### View Test Reports

```
📈 Test Reports
Available reports (showing last 10):
 1. 2024-01-15_14-30-22 (15.2MB)
 2. 2024-01-15_10-15-08 (12.8MB)
 3. 2024-01-14_16-45-33 (14.1MB)
 ...
11. 📁 Open reports directory
12. 🧹 Clean old reports
```

#### **Funcionalidades:**

##### **Selección de Reporte Individual**
Al seleccionar un reporte, muestra:

```
📊 Reports in 2024-01-15_14-30-22:
1. 📄 Test Report: pytest_report.html
2. 📊 Coverage Report: htmlcov/index.html
3. 📁 Open report directory
```

##### **Auto-apertura de Archivos**
- **Windows**: Usa `os.startfile()`
- **macOS**: Usa `open`
- **Linux**: Usa `xdg-open`

##### **Tipos de Reportes**
- **📄 pytest_report.html**: Reporte principal con resultados detallados
- **📊 htmlcov/index.html**: Reporte visual de coverage
- **📄 coverage.xml**: Coverage en formato XML para CI/CD
- **📄 junit.xml**: Resultados en formato JUnit
- **📄 test_output.txt**: Output completo de la ejecución

## 🔧 Configuración Avanzada

### Test Configuration Menu

```
🔧 Test Configuration
1. 📊 View current configuration
2. 🔧 Edit pytest.ini
3. 🔧 Edit conftest.py
4. 📁 View test directory structure
5. 🔧 Environment check
```

#### **Funcionalidades Detalladas:**

##### **1. 📊 View Current Configuration**
Muestra información del sistema:
```
📊 Current Test Configuration:
========================================
Project Root: /path/to/IATrading
Test Directory: /path/to/IATrading/tests
Reports Directory: /path/to/reports
Current Report: /path/to/current_report

📄 pytest.ini:
----------------------------------------
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
```

##### **2. 🔧 Edit pytest.ini**
Abre el archivo de configuración de pytest en el editor del sistema.

**Editor por defecto:**
- **Windows**: `notepad`
- **Unix/Linux**: `nano`
- **Custom**: Variable de entorno `$EDITOR`

##### **3. 🔧 Edit conftest.py**
Abre el archivo de fixtures compartidas.

##### **4. 📁 View Directory Structure**
Muestra árbol visual del directorio de tests:
```
📁 Test Directory Structure:
==================================
tests/
├── test_menu.py
├── pytest.ini
├── conftest.py
├── unit/
│   ├── test_core_comprehensive.py
│   ├── test_core_additional.py
│   └── __init__.py
├── integration/
│   ├── test_system_integration.py
│   └── __init__.py
└── reports/
    ├── latest/
    └── 2024-01-15_14-30-22/
```

##### **5. 🔧 Environment Check**
Verifica que todas las dependencias estén instaladas:
```
🔧 Environment Check
==================================
🐍 Python: 3.11.2
🧪 pytest 7.4.0
📊 coverage 7.2.5

📁 Test Files:
   Unit tests: 2
     • test_core_comprehensive.py
     • test_core_additional.py
```

### 📋 Test Status & Info

```
📋 Test Status & Information
============================================
📁 Project Root: /path/to/IATrading
🧪 Test Directory: /path/to/tests
📊 Reports Directory: /path/to/reports
📈 Current Report: /path/to/current_report

📊 Test Files:
   🎯 Unit Tests: 2
   🔗 Integration Tests: 1
   ⚡ Performance Tests: 0

📈 Recent Reports: 3
   📊 2024-01-15_14-30-22 (15.2MB)
   📊 2024-01-15_10-15-08 (12.8MB)
   📊 2024-01-14_16-45-33 (14.1MB)
```

## 🎯 Casos de Uso Recomendados

### 🔧 Workflow de Desarrollo Diario

```
Mañana:
1. Opción 1 → 5 (All Unit Tests Quick) - Verificar estado
2. Desarrollar código...
3. Opción 1 → 8 (Custom) - Test específico en desarrollo
4. Opción 1 → 6 (Verbose) - Verificar detalles

Tarde:
5. Opción 1 → 7 (Coverage) - Verificar cobertura
6. Opción 5 → 1 (Clean cache) - Si hay problemas
```

### 🚀 Workflow de Release

```
Pre-Release:
1. Opción 4 → 3 (Complete + Coverage) - Validación completa
2. Opción 6 (View Reports) - Revisar resultados
3. Opción 4 → 4 (+ Performance) - Si hay cambios de performance

Post-Release:
4. Opción 5 → 2 (Clean old reports) - Limpieza
5. Opción 8 (Status) - Documentar estado final
```

### 🐛 Workflow de Debugging

```
Cuando algo falla:
1. Opción 1 → 8 (Custom) - Test específico que falla
2. Opción 7 → 5 (Environment check) - Verificar entorno
3. Opción 6 (View Reports) - Comparar con reportes anteriores
4. Opción 5 → 4 (Reset test data) - Si problema persiste
5. Opción 1 → 6 (Verbose) - Re-ejecutar con detalles
```

### 📊 Workflow de Análisis de Performance

```
Optimización:
1. Opción 3 → 1 (Strategy Performance) - Baseline actual
2. Hacer cambios de optimización...
3. Opción 3 → 1 (Strategy Performance) - Nuevo benchmark
4. Opción 3 → 4 (Benchmark Comparison) - Comparar resultados
5. Opción 6 (View Reports) - Análisis visual
```

## 🔧 Troubleshooting

### Problemas Comunes

#### **❌ "pytest not found"**
**Solución:**
```bash
pip install pytest pytest-html pytest-cov
python -m pytest --version  # Verificar instalación
```

#### **❌ "Permission denied creating symlink"**
**Solución Windows:**
- Ejecutar como administrador
- O usar junction: `mklink /J`
- El menu usa fallback automático

#### **❌ "Tests not discovered"**
**Verificar:**
1. Archivos empiezan con `test_`
2. Funciones empiezan con `test_`
3. Clases empiezan con `Test`
4. Archivos tienen `__init__.py`

#### **❌ "Coverage report empty"**
**Solución:**
```bash
# Verificar que src/ existe y tiene código
ls -la src/
# Reinstalar coverage
pip install --upgrade coverage
```

#### **❌ "Cannot open HTML report"**
**Verificar:**
1. Archivo existe en `reports/latest/pytest_report.html`
2. Browser por defecto configurado
3. Permisos de archivo correctos

### Logs de Debugging

#### **Ubicación de Logs:**
- **Output completo**: `reports/TIMESTAMP/test_output.txt`
- **Logs específicos**: `reports/TIMESTAMP/logs/`
- **Coverage XML**: `reports/TIMESTAMP/coverage.xml`
- **JUnit XML**: `reports/TIMESTAMP/junit.xml`

#### **Formato de Logs:**
```
Test: Strategy Performance Tests
Timestamp: 2024-01-15 14:30:22
Command: python -m pytest unit/test_strategies.py::TestStrategyPerformance -v

STDOUT:
================================ test session starts ================================
...

STDERR:
...
```

### Performance Tuning

#### **Tests Lentos:**
```bash
# Identificar tests lentos
pytest --durations=10

# Ejecutar solo tests rápidos
pytest -m "not slow"

# Paralelización (si disponible)
pytest -n auto
```

#### **Reportes Grandes:**
```bash
# Limitar tamaño de reporte HTML
pytest --html=report.html --self-contained-html

# Comprimir reportes antiguos
gzip reports/*/pytest_report.html
```

## 📚 Referencias Adicionales

### Documentos Relacionados
- [Test Commands Guide](commands.md) - Comandos detallados de pytest
- [Coverage Reports](coverage.md) - Análisis de cobertura
- [CI/CD Integration](cicd.md) - Integración continua

### Enlaces Útiles
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pytest-HTML Plugin](https://pytest-html.readthedocs.io/)

### Configuración Avanzada

#### **pytest.ini Recomendado:**
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
    --strict-config
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    unit: marks tests as unit tests
```

#### **conftest.py Recomendado:**
```python
import pytest
import pandas as pd
from datetime import datetime
from src.core.types import TradingData

@pytest.fixture
def sample_trading_data():
    """Standard trading data for tests"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100,
        'close': [100.5] * 100,
        'volume': [1000] * 100
    }, index=dates)
    
    return TradingData(
        symbol='TEST',
        timeframe='H1',
        data=data,
        provider='test',
        timestamp=datetime.now()
    )

@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'test_mode': True,
        'debug': False,
        'timeout': 30
    }
```

---

**¡Con esta documentación tienes todo lo necesario para dominar el Test Menu Interactivo!** 🚀

El Test Menu es tu herramienta central para un testing eficiente y organizado en IATrading.