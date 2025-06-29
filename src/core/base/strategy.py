"""Unified BaseStrategy with all functionality"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import json

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from ...utils.logger import Logger

from ..types import TradingData, SignalData
from ..exceptions import StrategyError


class BaseStrategy(ABC):
    """
    Unified base class for all trading strategies.
    Combines validation, persistence, optimization, and ML features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with configuration"""
        self.strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        
        # Load configuration (from JSON file or provided config)
        self.config = self._load_strategy_config()
        if config:
            self.config.update(config)
        
        # Initialize parameters
        self.params = self.config.get('params', self.get_default_parameters())
        
        # Validate configuration
        self.validate_parameters(self.params)
        
        self.logger.debug(f"{self.strategy_name.upper()} strategy initialized with config: {self.params}")

    # ===========================================
    # CORE ABSTRACT METHODS (must implement)
    # ===========================================
    
    @abstractmethod
    def calculate_indicators(self, data: TradingData) -> pd.DataFrame:
        """Calculate technical indicators"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: TradingData) -> SignalData:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the strategy"""
        self.logger.info(f"{self.strategy_name.upper()} strategy initialized with config defaults")
        pass

    # ===========================================
    # CONFIGURATION & PERSISTENCE
    # ===========================================
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy configuration from JSON file"""
        config_path = Path(__file__).parent.parent.parent / 'strategies' / self.strategy_name / f"config_{self.strategy_name}.json"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.logger.debug(f"Loaded config from {config_path}")
                    return config
            else:
                # Create default configuration
                default_config = {
                    "params": self.get_default_parameters(),
                    "param_space": self.get_param_space(),
                    "optimized_params": {},
                    "backtest_results": []
                }
                
                # Ensure directory exists
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save default config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                
                self.logger.info(f"Created default config at {config_path}")
                return default_config
                
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return {
                "params": self.get_default_parameters(),
                "param_space": {},
                "optimized_params": {},
                "backtest_results": []
            }
    
    def save_config(self) -> None:
        """Save current configuration to JSON file"""
        config_path = Path(__file__).parent.parent.parent / 'strategies' / self.strategy_name / f"config_{self.strategy_name}.json"
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    # ===========================================
    # PARAMETER OPTIMIZATION
    # ===========================================
    
    def get_optimized_params(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get optimized parameters for specific symbol and timeframe"""
        key = f"{symbol}_{timeframe}"
        return self.config.get('optimized_params', {}).get(key, self.get_default_parameters())
    
    def save_optimized_params(self, params: Dict[str, Any], symbol: str, timeframe: str, 
                            performance_metrics: Dict[str, float] = None) -> None:
        """Save optimized parameters for specific symbol and timeframe"""
        key = f"{symbol}_{timeframe}"
        
        if 'optimized_params' not in self.config:
            self.config['optimized_params'] = {}
        
        optimization_result = {
            'params': params,
            'timestamp': pd.Timestamp.now().isoformat(),
            'performance': performance_metrics or {}
        }
        
        self.config['optimized_params'][key] = optimization_result
        self.save_config()
        
        self.logger.info(f"Saved optimized params for {key}: {params}")
    
    def get_param_space(self) -> Dict[str, List]:
        """Get parameter space for optimization (override in subclasses)"""
        # Default implementation - override in specific strategies
        defaults = self.get_default_parameters()
        param_space = {}
        
        for param, default_value in defaults.items():
            if isinstance(default_value, (int, float)):
                if param.endswith('_period') or param.endswith('_window'):
                    # Period parameters
                    param_space[param] = list(range(max(2, int(default_value * 0.5)), 
                                                  int(default_value * 2) + 1))
                elif param.endswith('_threshold'):
                    # Threshold parameters
                    param_space[param] = [default_value * i for i in [0.5, 0.75, 1.0, 1.25, 1.5]]
                else:
                    # Generic numeric parameters
                    param_space[param] = [default_value * i for i in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
        
        return param_space

    # ===========================================
    # VALIDATION METHODS
    # ===========================================
    
    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters (override in subclasses for specific validation)"""
        if not isinstance(params, dict):
            raise StrategyError("Parameters must be a dictionary")
        
        # Basic validation - override in subclasses for specific rules
        for key, value in params.items():
            if value is None:
                raise StrategyError(f"Parameter '{key}' cannot be None")
            
            # --- MODIFICADO: Excluye bool de la validación numérica ---
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
                if not key.endswith('_threshold'):
                    raise StrategyError(f"Parameter '{key}' must be positive, got {value}")
    
    def validate_data(self, data: TradingData) -> bool:  # ✅ ADD BOOL RETURN TYPE
        """Validate input data"""
        if data is None:
            raise StrategyError("Data cannot be None")
        
        if data.data.empty:
            raise StrategyError("Data cannot be empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.data.columns]
        if missing_columns:
            raise StrategyError(f"Data missing required columns: {missing_columns}")
        
        # Check for minimum required periods
        min_periods = self.get_required_periods()
        if len(data.data) < min_periods:
            raise StrategyError(f"Insufficient data: need at least {min_periods} periods, got {len(data.data)}")
        
        return True  # ✅ RETURN TRUE IF VALIDATION PASSES
    
    def get_required_periods(self) -> int:
        """Get minimum required periods for strategy (override in subclasses)"""
        # Default implementation - look for period parameters
        defaults = self.get_default_parameters()
        periods = []
        
        for key, value in defaults.items():
            if isinstance(value, int) and ('period' in key.lower() or 'window' in key.lower()):
                periods.append(value)
        
        return max(periods) if periods else 20  # Default minimum

    # ===========================================
    # DATA PROCESSING
    # ===========================================
    
    def preprocess_data(self, data: TradingData) -> TradingData:
        """Preprocess data before indicator calculation (override if needed)"""
        # Basic preprocessing - remove NaN values, ensure proper data types
        cleaned_data = data.data.dropna()
        
        # Ensure numeric columns are float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        return TradingData(
            symbol=data.symbol,
            timeframe=data.timeframe,
            data=cleaned_data,
            provider=data.provider,
            timestamp=data.timestamp  # ✅ ADD MISSING TIMESTAMP
        )
    
    def postprocess_signals(self, signals: SignalData) -> SignalData:
        """Postprocess signals after generation (override if needed)"""
        # Basic postprocessing - ensure signals are in valid range
        processed_signals = signals.signals.clip(-1, 1)
        
        return SignalData(
            signals=processed_signals,
            metadata=signals.metadata,
            strategy_name=signals.strategy_name,
            timestamp=signals.timestamp
        )

    # ===========================================
    # BACKTESTING & EVALUATION
    # ===========================================
    
    def backtest(self, data: TradingData, initial_capital: float = 10000,
                commission: float = 0.001) -> Dict[str, Any]:
        """Simple backtesting implementation"""
        try:
            # Validate and preprocess data
            self.validate_data(data)
            processed_data = self.preprocess_data(data)
            
            # Generate signals
            signal_data = self.generate_signals(processed_data)
            signals = signal_data.signals
            
            # Calculate returns
            prices = processed_data.data['close']
            returns = prices.pct_change()
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # Count trades
            position_changes = signals.diff().abs()
            num_trades = position_changes.sum() / 2  # Each trade has entry and exit
            
            results = {
                'total_return': round(total_return * 100, 2),
                'annual_return': round(annual_return * 100, 2),
                'volatility': round(volatility * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_drawdown * 100, 2),
                'num_trades': int(num_trades),
                'signal_changes': int(signal_data.metadata.get('signal_changes', 0)),
                'final_capital': round(initial_capital * (1 + total_return), 2),
                'backtest_period': f"{processed_data.data.index[0]} to {processed_data.data.index[-1]}"
            }
            
            # Save backtest results
            backtest_result = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'params': self.params.copy(),
                'results': results,
                'symbol': data.symbol,
                'timeframe': data.timeframe
            }
            
            if 'backtest_results' not in self.config:
                self.config['backtest_results'] = []
            
            self.config['backtest_results'].append(backtest_result)
            
            # Keep only last 10 backtest results
            self.config['backtest_results'] = self.config['backtest_results'][-10:]
            self.save_config()
            
            self.logger.info(f"Backtest completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise StrategyError(f"Backtest failed: {e}")

    def backtest_simple(
    self,
    main_data: TradingData,
    signals: Optional[pd.Series] = None,
    context_data: Optional[Dict[str, TradingData]] = None,
    initial_balance: float = 10000.0,
    commission: float = 0.0,          # comisión por operación (en %)
    slippage: float = 0.0,            # slippage por operación (en %)
    stop_loss: Optional[float] = None,    # stop-loss en % (ej: 0.02 para 2%)
    take_profit: Optional[float] = None   # take-profit en % (ej: 0.04 para 4%)
) -> dict:
        """
        Backtest robusto con soporte opcional para comisión, slippage, stop-loss y take-profit.
        """
        self.validate_data(main_data)
        df = main_data.data.copy()
        if signals is None:
            signals = self.generate_signals(main_data, context_data=context_data).signals

        df['signal'] = signals
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        df['returns'] = df['close'].pct_change().fillna(0)
        df['trade'] = df['position'].diff().fillna(0).abs() > 0

        # Aplica comisión y slippage solo en los cambios de posición
        trade_cost = commission + slippage
        df['cost'] = 0.0
        df.loc[df['trade'], 'cost'] = trade_cost

        equity = initial_balance
        equity_curve = []
        in_position = 0
        entry_price = 0.0
        num_trades = 0

        for idx, row in df.iterrows():
            if row['trade'] and row['position'] != 0:
                # Nueva entrada
                in_position = row['position']
                entry_price = row['close']
                equity -= equity * trade_cost  # Aplica comisión+slippage al entrar
                num_trades += 1

            if in_position != 0:
                # Retorno flotante
                ret = (row['close'] - entry_price) / entry_price * in_position
                stop_hit = take_hit = False
                if stop_loss is not None and ret <= -abs(stop_loss):
                    ret = -abs(stop_loss)
                    stop_hit = True
                if take_profit is not None and ret >= abs(take_profit):
                    ret = abs(take_profit)
                    take_hit = True
                if stop_hit or take_hit:
                    equity *= (1 + ret)
                    equity -= equity * trade_cost
                    in_position = 0
                    entry_price = 0.0
                elif row['trade'] and row['position'] == 0:
                    equity *= (1 + ret)
                    equity -= equity * trade_cost
                    in_position = 0
                    entry_price = 0.0
            equity_curve.append(equity)

        # --- FORZAR CIERRE DE POSICIÓN ABIERTA AL FINAL ---
        if in_position != 0 and entry_price != 0:
            ret = (df['close'].iloc[-1] - entry_price) / entry_price * in_position
            equity *= (1 + ret)
            equity_curve[-1] = equity
            self.logger.debug(f"\n{'='*60}\n⚡ [CIERRE FORZADO AL FINAL] ⚡\nRetorno: {ret:.4f} | Equity final: {equity:.2f}\n{'='*60}")

        df['equity'] = equity_curve
        self.logger.debug(f"\n{'*'*40}\n🔔 [RESUMEN BACKTEST] 🔔\nNúmero de trades: {num_trades}\nEquity final: {df['equity'].iloc[-1]:.2f}\n{'*'*40}")
        self.logger.debug(f"\n{'-'*40}\n🗓️ [PERIODO EVALUADO] 🗓️\n{df.index[0]} a {df.index[-1]}\n{'-'*40}")

        # Resultado del backtest
        result = {
            'final_equity': equity,
            'total_return': (equity / initial_balance) - 1,
            'num_trades': num_trades,
            'equity_curve': equity_curve,
            'df': df,
            'period': (df.index[0], df.index[-1])  # <-- Devuelve el periodo evaluado
        }
        return result