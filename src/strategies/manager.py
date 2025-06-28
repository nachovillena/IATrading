"""Strategy management and execution"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import importlib
import os

from ..core.types import TradingData, SignalData
from ..core.exceptions import StrategyError
from ..utils.logger import Logger

class StrategyManager:
    """Manages strategy loading and execution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy manager"""
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.strategies = {}
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Auto-discover and load available strategies from src/strategies/*/strategy.py"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        strategies_dir = base_dir
        for entry in os.listdir(strategies_dir):
            strategy_path = os.path.join(strategies_dir, entry)
            if os.path.isdir(strategy_path) and os.path.exists(os.path.join(strategy_path, "strategy.py")):
                strategy_name = entry
                module_path = f"src.strategies.{strategy_name}.strategy"
                class_name = f"{strategy_name.capitalize()}Strategy"
                # Carga config si existe
                config = self.load_strategy_config(strategy_name)
                self.logger.debug(f"Config cargado para {strategy_name}: {config}")
                try:
                    module = importlib.import_module(module_path)
                    strategy_class = getattr(module, class_name)
                    strategy_instance = strategy_class(config)
                    self.strategies[strategy_name] = strategy_instance
                    self.logger.info(f"✅ Loaded strategy: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"❌ Error loading strategy {strategy_name}: {e}")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategies.keys())
    
    def get_strategy(self, name: str):
        """Get strategy instance by name"""
        if name not in self.strategies:
            raise StrategyError(f"Strategy '{name}' not found. Available: {list(self.strategies.keys())}")
        return self.strategies[name]
    
    def process_signals(self, data: TradingData, strategy_names: List[str]) -> Dict[str, SignalData]:
        """Process signals for multiple strategies"""
        results = {}
        
        for strategy_name in strategy_names:
            try:
                self.logger.info(f"Processing strategy: {strategy_name}")
                
                # Get strategy instance
                strategy = self.get_strategy(strategy_name)
                
                # Validate data
                if not strategy.validate_data(data):
                    raise StrategyError(f"Invalid data for strategy {strategy_name}")
                
                # Preprocess data if needed
                processed_data = strategy.preprocess_data(data)
                
                # Generate signals
                signal_data = strategy.generate_signals(processed_data)
                
                # Store results
                results[strategy_name] = signal_data
                
                self.logger.info(f"✅ Successfully processed {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {strategy_name}: {e}")
                # Create empty signal data for failed strategies
                import pandas as pd
                results[strategy_name] = SignalData(
                    signals=pd.Series(dtype=int),
                    metadata={'error': str(e), 'last_signal': 0},
                    strategy=strategy_name,
                    timestamp=pd.Timestamp.now()
                )
        
        return results
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get information about a specific strategy"""
        try:
            strategy = self.get_strategy(strategy_name)
            return strategy.get_strategy_info()
        except Exception as e:
            return {'error': str(e)}
    
    def get_all_strategies_info(self) -> Dict[str, Any]:
        """Get information about all available strategies"""
        info = {}
        for strategy_name in self.get_available_strategies():
            info[strategy_name] = self.get_strategy_info(strategy_name)
        return info
    
    def get_required_timeframes(self, strategy_name: str) -> list:
        """
        Devuelve una lista con el timeframe principal y todas las temporalidades de contexto requeridas por la estrategia,
        leyendo los campos 'main_timeframe' y 'context_timeframes' de su configuración JSON.
        Si no están definidos, usa ['H1'] por defecto.
        """
        config = self.load_strategy_config(strategy_name)
        main_tf = config.get("main_timeframe", "H1")
        context_tfs = config.get("context_timeframes", ["H1"])
        # Unir y eliminar duplicados manteniendo el orden
        all_tfs = [main_tf] + [tf for tf in context_tfs if tf != main_tf]
        return all_tfs

    def load_strategy_config(self, strategy_name: str) -> dict:
        """
        Carga la configuración JSON de la estrategia indicada.
        """
        import os
        import json

        # Asume que las configs están en src/strategies/{strategy_name}/config_{strategy_name}.json
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, strategy_name, f"config_{strategy_name}.json")
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)