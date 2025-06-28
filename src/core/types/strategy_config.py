from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

@dataclass
class StrategyConfig:
    """Configuración para estrategias de trading"""
    name: str
    parameters: Dict[str, Any]
    enabled: bool = True
    risk_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")
        
        if self.risk_params is None:
            self.risk_params = {}