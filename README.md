# 🚀 Sistema de Trading IA - Arquitectura Modular

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Architecture](https://img.shields.io/badge/Architecture-Modular-green.svg)](/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](/)

## 📋 Descripción

Sistema de trading automatizado con inteligencia artificial que ha sido completamente refactorizado para ofrecer una **arquitectura modular, escalable y mantenible**. Incluye estrategias de trading avanzadas, optimización de hiperparámetros con ML, y múltiples interfaces de usuario.

## ✨ Características Principales

- 🧠 **Machine Learning Integrado**: Optimización bayesiana de hiperparámetros
- 📊 **Múltiples Estrategias**: EMA, RSI, MACD, Bollinger Bands, SMA
- 🔄 **Pipeline de Datos Robusto**: Control de calidad automático y cache inteligente
- 🎯 **Sistema de Señales Avanzado**: Agregación y filtrado de señales
- 💻 **Interfaces Múltiples**: CLI moderno, menú interactivo, web (futuro)
- 📈 **Análisis de Rendimiento**: Métricas completas y backtesting
- 🔧 **Arquitectura Modular**: Fácil extensión y mantenimiento

## 🏗️ Arquitectura del Sistema

```
IATrading/
├── 📁 src/                     # Código fuente principal
│   ├── 🧱 core/               # Clases base y tipos
│   ├── 📊 data/               # Pipeline de datos y proveedores
│   ├── 🎯 trading/            # Estrategias y señales
│   ├── 🤖 ml/                 # Machine Learning y optimización
│   ├── 🎼 services/           # Servicios de orquestación
│   └── 💻 interfaces/         # Interfaces de usuario
├── 📁 config/                 # Configuraciones
├── 📁 data/                   # Datos de trading
├── 📁 docs/                   # Documentación
├── 📁 tests/                  # Tests unitarios
└── 📄 Scripts principales     # Scripts de demo y validación
```

### 🧱 Módulos Principales

| Módulo | Descripción | Archivos Clave |
|--------|-------------|----------------|
| **core** | Clases base y tipos de datos | `base_classes.py`, `types.py`, `exceptions.py` |
| **data** | Pipeline de datos y proveedores | `pipeline.py`, `providers.py`, `quality.py` |
| **trading** | Estrategias y gestión de señales | `strategies.py`, `signals.py`, `portfolio.py` |
| **ml** | Modelos y optimización ML | `models.py`, `optimization.py`, `features.py` |
| **services** | Orquestación y configuración | `orchestrator.py`, `config_service.py` |
| **interfaces** | Interfaces de usuario | `cli_interface.py`, `menu_interface.py` |

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd IATrading

# Instalar dependencias
pip install -r requirements.txt

# Configurar entorno
python -c "import src; print('✅ Sistema listo')"
```

### 2. Verificación del Sistema

```bash
# Ejecutar demo completo
python demo_advanced.py

# Ejecutar tests
python test_comprehensive.py

# Validar migración
python validate_migration.py
```

### 3. Uso Básico

#### Menú Interactivo (NUEVO)
```bash
# Ejecutar el nuevo menú interactivo modular
python menu_app.py
```

#### CLI (Línea de Comandos)
```bash
# Ver estado del sistema
python -m src.interfaces.cli_interface status

# Ejecutar pipeline completo
python -m src.interfaces.cli_interface pipeline --symbol EURUSD

# Optimizar estrategia
python -m src.interfaces.cli_interface optimize --strategy ema --symbol EURUSD
```

#### Limpieza de Archivos Obsoletos
```bash
# Eliminar archivos y directorios obsoletos
python cleanup.py
```

#### Programático (Python)
```python
from src.services.orchestrator import TradingOrchestrator

# Crear orquestador
orchestrator = TradingOrchestrator()

# Ejecutar pipeline completo
result = orchestrator.run_full_pipeline('EURUSD', strategies=['ema', 'rsi'])

# Obtener estado del sistema
status = orchestrator.get_system_status()
```

## 📊 Ejemplos de Uso

### Estrategia Simple
```python
from src.trading.strategies import StrategyManager

# Crear manager de estrategias
manager = StrategyManager()

# Crear estrategia EMA
ema_strategy = manager.create_strategy('ema', {
    'ema_fast': 12,
    'ema_slow': 26,
    'signal_threshold': 0.001
})

# Generar señales
signals = ema_strategy.generate_signals(data)
```

### Pipeline de Datos
```python
from src.data.pipeline import DataPipeline

# Crear pipeline
pipeline = DataPipeline()

# Obtener datos con control de calidad
data = pipeline.fetch_data("EURUSD", "H1", source="yahoo")

# Verificar calidad
quality_report = pipeline.quality_controller.validate_data(data)
```

### Optimización ML
```python
from src.ml.optimization import HyperparameterOptimizer

# Crear optimizador
optimizer = HyperparameterOptimizer()

# Optimizar estrategia
result = optimizer.optimize_strategy(
    strategy='ema',
    symbol='EURUSD',
    n_trials=100
)

print(f"Mejores parámetros: {result['best_params']}")
print(f"Mejor score: {result['best_score']}")
```

## 🔧 Configuración

### Archivo Principal: `config/config.yaml`
```yaml
data:
  sources: ['yahoo', 'file']
  cache_enabled: true
  cache_ttl: 3600

trading:
  default_strategies: ['ema', 'rsi']
  risk_management: true
  max_positions: 10

ml:
  default_model: 'logistic'
  optimization_trials: 100
  validation_split: 0.2
```

### Configuración Programática
```python
from src.services.config_service import ConfigurationService

config = ConfigurationService()
config.set_config('data.cache_enabled', True)
config.set_config('ml.optimization_trials', 200)
```

## 📈 Estrategias Disponibles

| Estrategia | Descripción | Parámetros Principales |
|------------|-------------|------------------------|
| **EMA** | Exponential Moving Average | `ema_fast`, `ema_slow` |
| **RSI** | Relative Strength Index | `rsi_period`, `oversold`, `overbought` |
| **MACD** | Moving Average Convergence Divergence | `fast_period`, `slow_period`, `signal_period` |
| **Bollinger** | Bollinger Bands | `period`, `std_dev` |
| **SMA** | Simple Moving Average | `sma_fast`, `sma_slow` |

### Agregar Nueva Estrategia
```python
from src.core.base_classes import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implementar lógica de la estrategia
        signals = []
        # ... lógica ...
        return pd.DataFrame(signals)

# Registrar en el manager
from src.trading.strategies import StrategyManager
StrategyManager.register_strategy('my_strategy', MyStrategy)
```

## 🧪 Testing

### Ejecutar Tests
```bash
# Tests completos
python test_comprehensive.py

# Tests específicos
python -m pytest tests/ -v

# Coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Estructura de Tests
```
tests/
├── test_core/          # Tests de módulos core
├── test_data/          # Tests de pipeline de datos
├── test_trading/       # Tests de estrategias
├── test_ml/           # Tests de ML
├── test_services/     # Tests de servicios
└── test_interfaces/   # Tests de interfaces
```

## 📊 Métricas y Monitoring

### Métricas Principales
- **Sharpe Ratio**: Rentabilidad ajustada por riesgo
- **Maximum Drawdown**: Pérdida máxima desde el pico
- **Win Rate**: Porcentaje de operaciones ganadoras
- **Profit Factor**: Ratio ganancia/pérdida
- **Sortino Ratio**: Rentabilidad ajustada por riesgo negativo

### Ejemplo de Evaluación
```python
from src.services.evaluation_service import EvaluationService

evaluator = EvaluationService()
metrics = evaluator.evaluate_strategy('ema', 'EURUSD')

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

## 🔄 Migración desde Sistema Legacy

Si vienes del sistema anterior, consulta la [Guía de Migración](MIGRATION_GUIDE.md) para una transición suave.

### Compatibilidad Legacy
```python
# Usar funciones legacy durante la transición
from src.migration_helper import (
    legacy_optimize_strategy,
    legacy_generate_signals
)

# Funciona como el sistema anterior
result = legacy_optimize_strategy('ema', 'EURUSD', 100)
```

## 📚 Documentación

- 📖 **[Guía Completa de Refactoring](REFACTORING_SUMMARY.md)**: Detalles técnicos de la refactorización
- 🔄 **[Guía de Migración](MIGRATION_GUIDE.md)**: Cómo migrar del sistema legacy
- 🧪 **[Scripts de Demo](demo_advanced.py)**: Ejemplos completos de uso
- ✅ **[Validación](validate_migration.py)**: Scripts de validación del sistema

## 🛠️ Desarrollo

### Estructura para Contribuir
```bash
# Fork del repositorio
git clone <your-fork>

# Crear rama para feature
git checkout -b feature/nueva-estrategia

# Desarrollar con tests
python test_comprehensive.py

# Commit y push
git commit -m "feat: agregar nueva estrategia XYZ"
git push origin feature/nueva-estrategia
```

### Estándares de Código
- **PEP 8**: Estilo de código Python
- **Type Hints**: Tipado estático
- **Docstrings**: Documentación de funciones
- **Tests**: Cobertura mínima 80%

## 🐛 Troubleshooting

### Problemas Comunes

#### ImportError
```bash
# Verificar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# O usar imports relativos
python -m src.interfaces.cli_interface
```

#### Configuración no encontrada
```python
# Verificar archivos de configuración
from src.services.config_service import ConfigurationService
config = ConfigurationService()
print(config.get_config_path())
```

#### Datos no disponibles
```bash
# Verificar estructura de datos
python -c "from src.data.pipeline import DataPipeline; DataPipeline().verify_data_structure()"
```

## 📞 Soporte

- 🐛 **Issues**: Reportar bugs en GitHub Issues
- 💡 **Features**: Solicitar nuevas características
- 📖 **Docs**: Consultar documentación en `/docs`
- 🧪 **Tests**: Ejecutar `python test_comprehensive.py`

## 🗺️ Roadmap

### ✅ Fase 1 - Refactoring (COMPLETADO)
- Arquitectura modular
- Separación de responsabilidades
- Interfaces múltiples
- Sistema de testing

### 🔄 Fase 2 - Extensión (EN PROGRESO)
- Más estrategias de trading
- Interfaz web completa
- API REST
- Integración con MT5

### 🎯 Fase 3 - Optimización (FUTURO)
- Performance optimization
- Distributed computing
- Real-time data feeds
- Advanced ML models

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Comunidad Python por las librerías utilizadas
- Desarrolladores de Optuna para optimización
- Comunidad de trading algorítmico

---

## 🚀 ¡Empieza Ahora!

```bash
# Clonar y ejecutar demo
git clone <repository>
cd IATrading
python demo_advanced.py
```

**¡Tu sistema de trading IA con arquitectura empresarial está listo!** 🎉
