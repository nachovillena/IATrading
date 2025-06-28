# 🧪 Guía Completa de Comandos de Testing

## 📋 Tabla de Contenidos

1. [Comandos Básicos](#comandos-básicos)
2. [Tests por Categoría](#tests-por-categoría)
3. [Tests Específicos](#tests-específicos)
4. [Coverage Testing](#coverage-testing)
5. [Tests Selectivos](#tests-selectivos)
6. [Workflows Recomendados](#workflows-recomendados)

## 🎯 Comandos Básicos

### Test Simple
```bash
python -m pytest
```
**📅 Cuándo usar:** Verificación rápida diaria
**⏱️ Tiempo:** ~2-3 segundos
**📊 Output:** Resumen básico

### Test Verbose
```bash
python -m pytest -v
```
**📅 Cuándo usar:** Ver detalles de cada test
**⏱️ Tiempo:** ~2-3 segundos
**📊 Output:** Lista detallada de todos los tests

### Test con Outputs
```bash
python -m pytest -s -v
```
**📅 Cuándo usar:** Debugging con prints
**⏱️ Tiempo:** Variable
**📊 Output:** Incluye prints del código

## 🎯 Tests por Categoría

### Tests Unitarios
```bash
python -m pytest tests/unit/ -v
```
**📅 Cuándo usar:**
- ✅ Desarrollo de nuevas funciones
- ✅ Refactoring de código
- ✅ Tests rápidos durante desarrollo
- ✅ Debugging de componentes específicos

**📊 Incluye:**
- Core types testing
- Strategy validation
- Service integration
- Error handling

### Tests de Integración
```bash
python -m pytest tests/integration/ -v
```
**📅 Cuándo usar:**
- ✅ Antes de releases
- ✅ Verificar componentes trabajando juntos
- ✅ Testing end-to-end
- ✅ Validar workflows completos

**📊 Incluye:**
- System integration
- Data provider integration
- Strategy integration
- Performance testing
- Error handling

## 🎯 Tests Específicos

### Core Comprehensive
```bash
python -m pytest tests/unit/test_core_comprehensive.py -v
```
**📅 Cuándo usar:**
- ✅ Cambios en `TradingData`, `SignalData`
- ✅ Modificaciones en strategies
- ✅ Validación de tipos core
- ✅ Testing de integridad de datos

**📊 Tests incluidos:**
- `TestCoreTypesComprehensive` (2 tests)
- `TestStrategyComprehensive` (4 tests)
- `TestDataIntegrity` (2 tests)
- `TestErrorHandling` (2 tests)
- `TestMemoryAndPerformance` (1 test)

### Core Additional
```bash
python -m pytest tests/unit/test_core_additional.py -v
```
**📅 Cuándo usar:**
- ✅ Cambios en servicios (`ConfigService`, `EvaluationService`)
- ✅ Modificaciones en orchestrator
- ✅ Testing de performance y memoria
- ✅ Validación de componentes adicionales

**📊 Tests incluidos:**
- `TestCoreComponentsCoverage` (5 tests)
- `TestErrorScenarios` (3 tests)
- `TestServiceIntegration` (3 tests)
- `TestDataPipelineComprehensive` (2 tests)
- `TestStrategyManager` (3 tests)
- `TestPerformanceAndMemory` (2 tests)
- `TestIntegrationWorkflows` (1 test)

### System Integration
```bash
python -m pytest tests/integration/test_system_integration.py -v
```
**📅 Cuándo usar:**
- ✅ Cambios en providers (`YahooProvider`)
- ✅ Modificaciones en orchestrator
- ✅ Testing de threading/concurrencia
- ✅ Validación de performance con datasets grandes

**📊 Tests incluidos:**
- `TestSystemIntegration` (4 tests)
- `TestDataProviderIntegration` (2 tests)
- `TestStrategyIntegration` (2 tests)
- `TestSystemPerformance` (2 tests)
- `TestErrorHandling` (2 tests)

## 🎯 Coverage Testing

### Coverage Básico
```bash
python -m pytest --cov=src --cov-report=term-missing -v
```
**📅 Cuándo usar:**
- ✅ Revisión semanal de cobertura
- ✅ Identificar código no testeado
- ✅ Verificación rápida de coverage

### Coverage HTML Completo
```bash
python -m pytest --cov=src --cov-report=html:tests/reports/coverage --cov-report=term-missing -v
```
**📅 Cuándo usar:**
- ✅ Revisiones de código profundas
- ✅ Presentaciones a stakeholders
- ✅ Análisis detallado de gaps

**📁 Output:** `tests/reports/coverage/index.html`

### Coverage con Umbral
```bash
python -m pytest --cov=src --cov-fail-under=35 -v
```
**📅 Cuándo usar:**
- ✅ CI/CD pipelines
- ✅ Garantizar calidad mínima
- ✅ Prevenir degradación

**⚠️ Nota:** Actualmente el umbral está en 35%

## 🎯 Tests Selectivos

### Por Patrón
```bash
# Solo tests de strategy
python -m pytest -k "strategy" -v

# Solo tests de integration
python -m pytest -k "integration" -v

# Solo tests de performance
python -m pytest -k "performance" -v

# Solo tests de error handling
python -m pytest -k "error" -v

# Solo tests de memory
python -m pytest -k "memory" -v
```

### Por Marcadores
```bash
# Solo integration tests
python -m pytest -m integration -v

# Excluir tests lentos
python -m pytest -m "not slow" -v
```

### Test Específico
```bash
# Un test específico
python -m pytest tests/unit/test_core_additional.py::TestStrategyManager::test_strategy_creation -v

# Una clase de tests
python -m pytest tests/unit/test_core_additional.py::TestStrategyManager -v
```

## 🎯 Workflows Recomendados

### 🔧 Development Workflow
```bash
# 1. Durante desarrollo (cada 30 min)
python -m pytest tests/unit/test_core_additional.py -v

# 2. Antes de commit
python -m pytest tests/unit/ -v

# 3. Antes de push
python -m pytest --cov=src --cov-fail-under=33 -v
```

### 🚀 Release Workflow
```bash
# 1. Validación completa
python -m pytest tests/unit/ tests/integration/ --cov=src --cov-report=html:tests/reports/final --cov-fail-under=35 -v

# 2. Performance validation
python -m pytest -k "performance or memory" -v

# 3. Integration validation
python -m pytest tests/integration/ -v
```

### 🐛 Debugging Workflow
```bash
# 1. Test específico que falla
python -m pytest tests/unit/test_core_additional.py::TestStrategyManager::test_strategy_creation -v -s

# 2. Tests relacionados a un componente
python -m pytest -k "orchestrator" -v -s

# 3. Solo ver errores
python -m pytest --tb=short -v
```

### 🔄 CI/CD Workflow
```bash
# Para pipelines automatizados
python -m pytest --cov=src --cov-fail-under=35 --tb=short -q
```

## 📊 Referencia Rápida

| **Situación** | **Comando** | **Frecuencia** |
|---------------|-------------|----------------|
| 🔧 **Desarrollo feature** | `pytest tests/unit/ -v` | Cada 30 min |
| 🐛 **Debug específico** | `pytest -k "test_name" -v -s` | Según necesidad |
| 📊 **Review coverage** | `pytest --cov=src --cov-report=html -v` | Semanal |
| 🚀 **Pre-release** | `pytest tests/ --cov=src --cov-fail-under=35 -v` | Antes de release |
| ⚡ **Quick check** | `pytest tests/unit/test_core_additional.py -v` | Múltiples/día |
| 🔄 **CI/CD** | `pytest --cov=src --cov-fail-under=35 --tb=short` | Cada push |

## 🎯 Tu Comando Ganador

Basado en tu setup exitoso:

```bash
# Para desarrollo diario
python -m pytest tests/unit/test_core_additional.py tests/integration/test_system_integration.py --cov=src --cov-report=term-missing -v

# Para validación completa
python -m pytest tests/ --cov=src --cov-report=html:tests/reports/coverage_final -v
```

## 📝 Notas Importantes

- ✅ **Todos los integration tests pasan** (12/12)
- ✅ **41 de 42 tests exitosos** (97.6% success rate)
- ✅ **Coverage actual: 35%**
- ⚠️ **1 test skipped** (ProviderFactory no disponible)
- 🔧 **Tiempo promedio:** 1-2 segundos para tests completos