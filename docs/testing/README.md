# 🧪 Testing Documentation

Documentación completa del sistema de testing para IATrading.

## 📋 Índice

- **[📟 Test Menu Interactivo](test-menu-guide.md)** - Guía completa del menú de testing
- **[📋 Guía de Comandos](commands.md)** - Cuándo usar cada comando
- **[🏗️ Arquitectura de Tests](architecture.md)** - Estructura y organización
- **[🎯 Mejores Prácticas](best-practices.md)** - Standards y convenciones
- **[📊 Coverage Reports](coverage.md)** - Análisis de cobertura
- **[🔧 Troubleshooting](troubleshooting.md)** - Solución de problemas
- **[🔄 CI/CD Integration](cicd.md)** - Integración continua

## 🚀 Quick Start

### Método Interactivo (Recomendado)
```bash
# Ejecutar menú interactivo
python tests/test_menu.py

# Opciones principales:
# 1. Unit Tests (desarrollo diario)
# 4. Full Test Suite (validación completa)
# 5. Clean Environment (mantenimiento)
```

### Método Directo
```bash
# Test básico
python -m pytest tests/unit/test_core_additional.py -v

# Test completo con coverage
python -m pytest tests/ --cov=src --cov-report=html:tests/reports/coverage -v

# Test específico
python -m pytest -k "strategy" -v
```

## 📊 Estado Actual

- **Total Tests**: 42
- **Coverage**: 35%
- **Success Rate**: 97.6%
- **Herramientas**: pytest, coverage, html reports
- **Last Update**: $(date)

## 🎯 Flujos Recomendados

### 🔧 Desarrollo Diario
1. **Test Menu** → Opción 1 → 5 (Unit Tests Quick)
2. **Desarrollo** → Test específicos
3. **Finalización** → Opción 1 → 7 (Coverage)

### 🚀 Pre-Release
1. **Test Menu** → Opción 4 → 3 (Complete + Coverage)
2. **Verificar reportes** → Opción 6
3. **Limpieza** → Opción 5

### 🐛 Debugging
1. **Test Menu** → Opción 1 → 8 (Custom)
2. **Análisis** → Opción 6 (View Reports)