"""Configuration loader for trading system constants"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from ...utils.logger import Logger

class ConfigLoader:
    """Loader for trading system configuration"""
    
    def __init__(self):
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self._config_cache: Optional[Dict[str, Any]] = None
        self._config_file = Path(__file__).parent / "constants.json"
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if self._config_cache is None or force_reload:
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    self._config_cache = json.load(f)
                self.logger.debug(f"Loaded config from {self._config_file}")
            except FileNotFoundError:
                self.logger.error(f"Config file not found: {self._config_file}")
                self._config_cache = self._get_default_config()
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in config file: {e}")
                self._config_cache = self._get_default_config()
        
        return self._config_cache
    
    def get_signals(self) -> Dict[str, Any]:
        """Get signal constants"""
        config = self.load_config()
        return config.get("signals", {})
    
    def get_timeframes(self) -> Dict[str, str]:
        """Get timeframe constants"""
        config = self.load_config()
        return config.get("timeframes", {})
    
    def get_symbols(self) -> Dict[str, Any]:
        """Get symbol constants"""
        config = self.load_config()
        return config.get("symbols", {})
    
    def get_optimization_metrics(self) -> list:
        """Get optimization metrics"""
        config = self.load_config()
        return config.get("optimization", {}).get("metrics", [])
    
    def get_trade_status(self) -> Dict[str, str]:
        """Get trade status constants"""
        config = self.load_config()
        return config.get("trade_status", {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not available"""
        return {
            "signals": {
                "SIGNAL_BUY": 1,
                "SIGNAL_HOLD": 0,
                "SIGNAL_SELL": -1,
                "SIGNAL_NAMES": {"1": "BUY", "0": "HOLD", "-1": "SELL"}
            },
            "timeframes": {"1d": "D1", "1h": "H1"},
            "symbols": {"forex_pairs": [], "major_stocks": []},
            "optimization": {"metrics": ["sharpe_ratio"]},
            "trade_status": {"OPEN": "Open", "CLOSED": "Closed"}
        }

# Singleton instance
_config_loader = ConfigLoader()

# Convenience functions
def get_signal_constants():
    """Get signal constants"""
    signals = _config_loader.get_signals()
    return (
        signals.get("SIGNAL_BUY", 1),
        signals.get("SIGNAL_HOLD", 0), 
        signals.get("SIGNAL_SELL", -1),
        signals.get("SIGNAL_NAMES", {})
    )

def get_timeframes():
    """Get timeframes"""
    return _config_loader.get_timeframes()

def get_forex_pairs():
    """Get forex pairs"""
    symbols = _config_loader.get_symbols()
    return symbols.get("forex_pairs", [])

def get_major_stocks():
    """Get major stocks"""
    symbols = _config_loader.get_symbols()
    return symbols.get("major_stocks", [])

def get_optimization_metrics():
    """Get optimization metrics"""
    return _config_loader.get_optimization_metrics()

def get_trade_status():
    """Get trade status"""
    return _config_loader.get_trade_status()

# Export constants for backward compatibility
SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_NAMES = get_signal_constants()
TIMEFRAMES = get_timeframes()
FOREX_PAIRS = get_forex_pairs()
MAJOR_STOCKS = get_major_stocks()
OPTIMIZATION_METRICS = get_optimization_metrics()
TRADE_STATUS = get_trade_status()