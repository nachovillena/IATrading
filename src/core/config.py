"""Configuration management"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from ..utils.logger import Logger

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Look for config files in order of preference
        possible_paths = [
            'config.yaml',
            'config.yml', 
            'config.json',
            'src/config/config.yaml',
            'src/config/config.yml',
            'src/config/config.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default yaml path if none found
        return 'config.yaml'
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith(('.yaml', '.yml')):
                        import yaml
                        config = yaml.safe_load(f) or {}
                    else:
                        config = json.load(f)
                
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                self.logger.info(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data_pipeline': {
                'cache': {
                    'enabled': True,
                    'cache_dir': 'data/cache',
                    'ttl_hours': 24
                },
                'quality': {
                    'min_records': 100,
                    'max_gap_hours': 24,
                    'required_columns': ['open', 'high', 'low', 'close', 'volume']
                },
                'providers': {
                    'yahoo': {
                        'timeout': 30,
                        'retry_attempts': 3
                    },
                    'mt5': {
                        'timeout': 30,
                        'login': None,
                        'password': None,
                        'server': None
                    }
                }
            },
            'strategies': {
                'ema': {
                    'ema_fast': 12,
                    'ema_slow': 26,
                    'signal_threshold': 0.001
                },
                'rsi': {
                    'rsi_period': 14,
                    'oversold': 30,
                    'overbought': 70
                },
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            },
            'ml': {
                'features': {
                    'technical_indicators': True,
                    'market_data': True,
                    'time_features': True
                },
                'models': {
                    'default_model': 'random_forest',
                    'hyperparameter_tuning': True
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/trading_system.log'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        
        except Exception as e:
            self.logger.error(f"Error getting config key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        try:
            keys = key.split('.')
            current = self._config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            
            self.logger.info(f"Set config {key} = {value}")
        
        except Exception as e:
            self.logger.error(f"Error setting config key {key}: {e}")
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        try:
            save_path = path or self.config_path
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith(('.yaml', '.yml')):
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config, f, indent=2)
            
            self.logger.info(f"Saved configuration to {save_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._config = self._load_config()
        self.logger.info("Configuration reloaded")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()

# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager