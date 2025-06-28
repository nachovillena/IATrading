# Sistema de Trading con IA - Guía del Usuario

## 🚀 Introducción

El Sistema de Trading con IA es una plataforma completa para análisis técnico y generación automática de señales de trading. Integra múltiples estrategias técnicas con capacidades de machine learning para optimizar decisiones de inversión.

## 🎯 Características Principales

### ✅ **Estrategias Implementadas**
- **EMA (Exponential Moving Average)**: Detecta cruces de medias móviles
- **RSI (Relative Strength Index)**: Identifica condiciones de sobrecompra/sobreventa
- **MACD (Moving Average Convergence Divergence)**: Analiza convergencia/divergencia

### ✅ **Fuentes de Datos**
- **Yahoo Finance**: Forex, acciones, commodities
- **MetaTrader 5**: Datos en tiempo real (próximamente)
- **Cache inteligente**: Optimiza velocidad y reduce llamadas API

### ✅ **Instrumentos Soportados**
- **Forex**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
- **Acciones**: AAPL, GOOGL, MSFT, AMZN, TSLA
- **Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d

## 🏁 Instalación y Configuración

### 1. Requisitos del Sistema
```bash
Python 3.8+
RAM: 4GB mínimo (8GB recomendado)
Espacio: 2GB para cache y datos
```

### 2. Instalación
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/IATrading.git
cd IATrading

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configuración Inicial
```bash
# Copiar configuración por defecto
cp config/config.example.yaml config/config.yaml

# Editar configuración según necesidades
notepad config/config.yaml  # Windows
```

## 🎮 Uso Básico

### 1. Interfaz de Línea de Comandos

#### Análisis Rápido
```bash
# Análisis simple de EURUSD con EMA
python -m src.cli analyze EURUSD --strategy ema --timeframe 1h

# Análisis con múltiples estrategias
python -m src.cli analyze GBPUSD --strategy ema,rsi,macd --days 90
```

#### Comandos Disponibles
```bash
# Ver estado del sistema
python -m src.cli status

# Listar estrategias disponibles  
python -m src.cli strategies list

# Limpiar cache
python -m src.cli cache clear

# Optimizar estrategia
python -m src.cli optimize EURUSD --strategy ema --metric sharpe
```

### 2. Interfaz de Menús Interactiva

```bash
# Ejecutar menú principal
python -m src.menu

# Opciones disponibles:
# 1. Análisis de instrumentos
# 2. Configuración de estrategias  
# 3. Optimización de parámetros
# 4. Visualización de resultados
# 5. Gestión de datos y cache
```

### 3. Uso Programático

#### Ejemplo Básico
```python
from src.services.orchestrator import TradingOrchestrator

# Crear orquestador
orchestrator = TradingOrchestrator()

# Análisis completo
result = orchestrator.run_full_pipeline(
    symbol='EURUSD',
    timeframe='1h',
    strategies=['ema', 'rsi'],
    period_days=90
)

# Revisar resultados
print(f"Estado: {result['status']}")
print(f"Señales generadas: {len(result['signals'])}")
print(f"Estrategias ejecutadas: {result['strategies_executed']}")
```

#### Análisis de Estrategia Individual
```python
from src.strategies.ema.strategy import EMAStrategy
from src.data.providers.yahoo import YahooProvider

# Obtener datos
provider = YahooProvider()
data = provider.fetch_data('EURUSD', '1h', period_days=30)

# Ejecutar estrategia
strategy = EMAStrategy()
signals = strategy.generate_signals(data)

# Analizar resultados
print(f"Última señal: {signals.signals.iloc[-1]}")
print(f"Tendencia: {signals.metadata['trend']}")
print(f"Cambios de señal: {signals.metadata['signal_changes']}")
```

## 📊 Interpretación de Resultados

### 1. Señales de Trading

| Valor | Significado | Acción Recomendada |
|-------|-------------|-------------------|
| **1** | BUY (Compra) | Considerar apertura de posición larga |
| **0** | HOLD (Mantener) | Mantener posición actual |
| **-1** | SELL (Venta) | Considerar apertura de posición corta |

### 2. Metadatos de Estrategias

#### EMA Strategy
```python
metadata = {
    'ema_fast': 1.0952,           # Valor EMA rápida actual
    'ema_slow': 1.0948,           # Valor EMA lenta actual  
    'current_diff_pct': 0.037,    # Diferencia porcentual
    'trend': 'bullish',           # Tendencia detectada
    'signal_changes': 23          # Número de cambios de señal
}
```

#### RSI Strategy
```python
metadata = {
    'current_rsi': 67.3,          # RSI actual (0-100)
    'market_condition': 'normal', # overbought/oversold/normal
    'rsi_period': 14,            # Período utilizado
    'signal_changes': 15         # Cambios de señal en el período
}
```

#### MACD Strategy
```python
metadata = {
    'current_macd': 0.0003,      # Valor MACD actual
    'current_signal': 0.0002,    # Línea de señal actual
    'current_histogram': 0.0001, # Histograma (MACD - Signal)
    'trend': 'bullish'          # Tendencia detectada
}
```

### 3. Sistema de Status

```python
system_status = {
    'status': 'healthy',          # healthy/warning/error
    'timestamp': '2024-06-25 15:30:00',
    'components': {
        'data': True,             # Providers funcionando
        'strategies': True,       # Estrategias cargadas
        'cache': True            # Cache operativo
    },
    'cache': {
        'hit_rate': 0.85,        # 85% de aciertos en cache
        'size_mb': 245.3,        # Tamaño actual del cache
        'entries': 1547          # Número de entradas
    },
    'strategies': ['ema', 'rsi', 'macd']  # Estrategias disponibles
}
```

## ⚙️ Configuración Avanzada

### 1. Personalización de Estrategias

#### EMA personalizada
```python
ema_config = {
    'ema_fast': 8,              # EMA rápida (default: 12)
    'ema_slow': 21,             # EMA lenta (default: 26)  
    'signal_threshold': 0.002   # Umbral señal (default: 0.001)
}

strategy = EMAStrategy(ema_config)
```

#### RSI personalizada
```python
rsi_config = {
    'rsi_period': 21,          # Período RSI (default: 14)
    'oversold': 25,            # Nivel sobreventa (default: 30)
    'overbought': 75           # Nivel sobrecompra (default: 70)
}

strategy = RSIStrategy(rsi_config)
```

### 2. Configuración de Cache

```yaml
# config/config.yaml
cache:
  enabled: true
  directory: "./cache"
  ttl_seconds: 3600          # 1 hora
  max_size_mb: 500          # 500MB máximo
  compression: true         # Compresión automática
```

### 3. Configuración de Providers

```yaml
providers:
  yahoo:
    timeout: 30             # Timeout en segundos
    retry_attempts: 3       # Reintentos en caso de error
    rate_limit: 100         # Llamadas por minuto
  
  mt5:
    server: "MetaQuotes-Demo"
    login: 12345
    password: "your_password"
```

## 📈 Casos de Uso Avanzados

### 1. Análisis Multi-Timeframe
```python
timeframes = ['15m', '1h', '4h', '1d']
results = {}

for tf in timeframes:
    result = orchestrator.run_full_pipeline(
        symbol='EURUSD',
        timeframe=tf,
        strategies=['ema', 'rsi', 'macd']
    )
    results[tf] = result

# Analizar confluencia entre timeframes
```

### 2. Optimización de Parámetros
```python
from src.ml.optimization import GridSearchOptimizer

# Definir rango de parámetros
param_grid = {
    'ema_fast': [8, 10, 12, 15],
    'ema_slow': [21, 26, 30, 35],
    'signal_threshold': [0.001, 0.002, 0.005]
}

# Ejecutar optimización
optimizer = GridSearchOptimizer()
best_params = optimizer.optimize(
    strategy='ema',
    symbol='EURUSD',
    param_grid=param_grid,
    metric='sharpe_ratio'
)
```

### 3. Backtesting Avanzado
```python
# Análisis histórico extenso
result = orchestrator.run_full_pipeline(
    symbol='EURUSD',
    timeframe='1h',
    period_days=365,        # 1 año de datos
    strategies=['ema', 'rsi', 'macd']
)

# Métricas de performance
performance = result['performance_metrics']
print(f"Sharpe Ratio: {performance['sharpe_ratio']}")
print(f"Max Drawdown: {performance['max_drawdown']}")
print(f"Win Rate: {performance['win_rate']}")
```

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. "No data available for symbol"
```bash
# Verificar conectividad
python -m src.cli test-connection

# Verificar símbolo válido
python -m src.cli symbols list --provider yahoo
```

#### 2. "Strategy not found"
```bash
# Listar estrategias disponibles
python -m src.cli strategies list

# Verificar instalación
python -m src.cli system check
```

#### 3. "Cache permission denied"
```bash
# Cambiar directorio de cache
export TRADING_CACHE_DIR=/tmp/trading_cache

# Limpiar cache existente
python -m src.cli cache clear
```

#### 4. Rendimiento lento
```bash
# Activar cache
python -m src.cli config set cache.enabled true

# Aumentar TTL del cache
python -m src.cli config set cache.ttl_seconds 7200  # 2 horas

# Verificar espacio en disco
df -h  # Linux/Mac
dir C:\ # Windows
```

### Logs y Debugging

#### Activar modo verbose
```bash
python -m src.cli analyze EURUSD --strategy ema --verbose
```

#### Ubicación de logs
```bash
# Logs del sistema
./logs/trading_system.log

# Logs de errores  
./logs/errors.log

# Logs de performance
./logs/performance.log
```

## 📞 Soporte

### Reportar Problemas
1. Verificar logs en `./logs/`
2. Ejecutar `python -m src.cli system check`
3. Crear issue en GitHub con información completa

### Recursos Adicionales
- **Documentación técnica**: `/docs/technical/`
- **Ejemplos**: `/examples/`
- **Tests**: `/tests/` para casos de uso
- **Wiki**: GitHub Wiki con tutoriales

### Configuración Recomendada por Perfil

#### Trader Principiante
```yaml
strategies:
  enabled: ['ema']
  ema:
    ema_fast: 12
    ema_slow: 26
timeframes: ['1h', '4h']
symbols: ['EURUSD', 'GBPUSD']
```

#### Trader Intermedio  
```yaml
strategies:
  enabled: ['ema', 'rsi']
timeframes: ['15m', '1h', '4h']
symbols: ['EURUSD', 'GBPUSD', 'USDJPY']
optimization: true
```

#### Trader Avanzado
```yaml
strategies:
  enabled: ['ema', 'rsi', 'macd']
timeframes: ['5m', '15m', '1h', '4h', '1d']
symbols: ['all_forex', 'major_stocks']
ml_features: true
advanced_optimization: true
```