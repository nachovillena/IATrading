from pydantic import BaseModel
from typing import List, Dict
import yaml

class Config(BaseModel):
    symbols: List[str]
    timeframes: List[str]
    parameters: Dict[str, list]
    data_paths: Dict[str, str]

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
