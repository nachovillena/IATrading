"""Configuration service for the trading system"""

from pathlib import Path
import yaml
from typing import Dict, Any, Optional
from ..core.exceptions import ConfigurationError

class ConfigService:
    """Service for managing system configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration service
        
        Args:
            config_path: Path to config.yaml file, defaults to src/config/config.yaml
        """
        self.config_path = Path(config_path or "src/config/config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration"""
        return self.config.get("data", {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-related configuration
        
        Returns:
            Dictionary containing trading settings including:
            - Default timeframe
            - Risk management settings
            - Position sizing rules
            - Trading hours
        """
        default_trading_config = {
            "default_timeframe": "H1",
            "default_symbol": "EURUSD",
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "max_daily_risk": 0.06,
                "stop_loss": 0.02,
                "take_profit": 0.04
            },
            "position_sizing": {
                "method": "fixed_fraction",
                "fraction": 0.02,
                "max_positions": 3
            },
            "trading_hours": {
                "start": "00:00",
                "end": "23:59",
                "timezone": "UTC"
            }
        }
        
        # Get trading config from file or use defaults
        trading_config = self.config.get("trading", {})
        
        # Merge with defaults
        return self._merge_with_defaults(trading_config, default_trading_config)

    def get_strategies_config(self) -> Dict[str, Any]:
        """Get strategies-related configuration"""
        return self.config.get("strategies", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging-related configuration"""
        return self.config.get("logging", {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/trading.log"
        })

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.config.get("paths", {})

    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a specific configuration value and save to file
        
        Args:
            section: Configuration section (e.g., 'data', 'trading')
            key: Configuration key to update
            value: New value
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value
            
            # Save updated config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, indent=2, allow_unicode=True)
                
        except Exception as e:
            raise ConfigurationError(f"Error updating configuration: {str(e)}")

    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration
        
        Returns:
            Dictionary containing ML settings including:
            - Model parameters
            - Feature engineering settings
            - Training configurations
        """
        default_ml_config = {
            "feature_engineering": {
                "lookback_period": 20,
                "technical_indicators": ["sma", "ema", "rsi", "macd", "bollinger"],
                "price_features": ["returns", "log_returns", "volatility"],
                "time_features": ["hour", "day_of_week", "month"]
            },
            "model_selection": {
                "default_model": "random_forest",
                "available_models": ["random_forest", "xgboost", "lstm", "svm"],
                "validation_method": "time_series_split",
                "test_size": 0.2
            },
            "hyperparameters": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2
                },
                "xgboost": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6
                }
            }
        }
        
        # Get ML config from file or use defaults
        ml_config = self.config.get("ml", {})
        
        # Merge with defaults
        return self._merge_with_defaults(ml_config, default_ml_config)

    def _merge_with_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration with defaults"""
        result = defaults.copy()
        
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_with_defaults(value, result[key])
            else:
                result[key] = value
                
        return result

    def get_system_config(self) -> Dict[str, Any]:
        """Get system-wide configuration
        
        Returns:
            Dictionary containing system settings including:
            - Basic system information
            - Performance settings
            - Monitoring configuration
            - Cleanup settings
            - Notification preferences
        """
        default_system_config = {
            "name": "IATrading",
            "version": "1.0.0",
            "environment": "development",
            "debug": False,
            "workers": 4,
            "max_memory": "2GB",
            "monitoring": {
                "enabled": True,
                "interval": 60,
                "metrics": ["cpu_usage", "memory_usage", "active_strategies"]
            },
            "cleanup": {
                "enabled": True,
                "interval": 86400,
                "keep_days": 30
            },
            "notifications": {
                "enabled": False,
                "methods": ["log"],
                "email": {
                    "smtp_server": "",
                    "port": 587,
                    "username": "",
                    "password": ""
                }
            }
        }
        
        # Get system config from file or use defaults
        system_config = self.config.get("system", {})
        
        # Merge with defaults to ensure all required settings exist
        return self._merge_with_defaults(system_config, default_system_config)
