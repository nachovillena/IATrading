"""Base data provider class"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path

from src import data
from ...utils.logger import Logger

from src.data.cache import get_global_cache
from ..types import TradingData
from ..exceptions import DataProviderError, ValidationError


class BaseProvider(ABC):
    """Base class for data providers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base provider
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        
        # Provider metadata
        self.provider_id = self.config.get('provider_id', self.name.lower())
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30)
        self.rate_limit = self.config.get('rate_limit', 1.0)  # seconds between requests
        
        # Cache settings
        self.use_cache = self.config.get('use_cache', True)
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data quality settings
        self.min_data_points = self.config.get('min_data_points', 100)
        self.max_missing_ratio = self.config.get('max_missing_ratio', 0.05)
        
        # Connection state
        self.is_connected = False
        self.last_request_time: Optional[datetime] = None
        
        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.cache_hit_count = 0
        
    def get_data(
        self,
        symbol: str,
        timeframe: str = 'M15',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_reload: bool = False,
        allow_download: bool = False,
        quality_check: Union[bool, Callable[[pd.DataFrame], dict]] = True,
        output_format: str = "dataframe",
        **kwargs
    ) -> TradingData:
        """
        Obtiene datos de la caché Parquet por símbolo y año. 
        Si allow_download=True y no hay datos, los descarga y guarda en caché.
        Si allow_download=False y no hay datos, lanza error.
        """
        cache = get_global_cache()
        if start_date is None or end_date is None:
            raise DataProviderError("Debes indicar start_date y end_date para la consulta en caché parquet.")

        cached_data = None
        if not force_reload:
            cached_data = cache.load(symbol, timeframe,start_date, end_date)

        if allow_download:
            # Descarga usando el método abstracto del provider concreto
            downloaded = self._download_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
            if downloaded is not None and not downloaded.empty:
                downloaded = self.ensure_required_columns(downloaded)
                self.validate_data_quality(downloaded)
                self.logger.info(f"Descargando {symbol} de {start_date} a {end_date} con {len(downloaded)} registros.")
                self.save_to_cache(symbol, timeframe, downloaded)
                cached_data = cache.load(symbol, timeframe, start_date, end_date)

        if cached_data is None or cached_data.empty:
            raise DataProviderError(f"No hay datos en caché para {symbol} entre {start_date} y {end_date}")

        # Selección de función de control de calidad
        if callable(quality_check):
            quality_fn = quality_check
        elif quality_check:
            quality_fn = self.validate_data_quality
        else:
            quality_fn = lambda df: {'quality_score': 1.0}

        quality_metrics = quality_fn(cached_data)

        # Salida en el formato deseado
        if output_format == "dataframe":
            output = cached_data
        elif output_format == "dict":
            output = cached_data.to_dict()
        elif output_format == "csv":
            output = cached_data.to_csv()
        else:
            output = cached_data

        return TradingData(
            symbol=symbol,
            timeframe=timeframe,
            data=output,
            provider=self.name,
            timestamp=datetime.now(),
            quality_score=quality_metrics.get('quality_score', 1.0)
    )

    @abstractmethod
    def _download_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Método que debe implementar cada provider concreto para descargar los datos nuevos.
        """
        pass

    def validate_symbol(self, symbol: str) -> bool:
        """Valida el formato del símbolo para el provider."""
        raise NotImplementedError

    def get_available_symbols(self) -> List[str]:
        """Devuelve una lista de símbolos soportados (si aplica)."""
        raise NotImplementedError

    def get_symbol_info(self, symbol: str) -> dict:
        """Devuelve metadatos del símbolo."""
        raise NotImplementedError
    
    def download_max_history(
        self,
        symbol: str,
        timeframe: str,
        force_reload: bool = False,
        **kwargs
    ) -> Any:
        """Descarga el máximo histórico posible para un símbolo/timeframe."""
        raise NotImplementedError

    def search_available_range(
        self,
        symbol: str,
        timeframe: str,
        **kwargs
    ) -> Optional[Dict[str, datetime]]:
        """Busca el rango de fechas disponible para un símbolo/timeframe."""
        raise NotImplementedError

    def _convert_timeframe(self, timeframe: str) -> Optional[str]:
        """Convierte un timeframe estándar al formato del provider."""
        raise NotImplementedError
    
    def connect(self) -> bool:
        """Connect to the data provider
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Default implementation - override in subclasses
            self.is_connected = True
            self.logger.info(f"Connected to {self.name}")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the data provider"""
        self.is_connected = False
        self.logger.info(f"Disconnected from {self.name}")
    
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            True if timeframe is valid, False otherwise
        """
        if not timeframe or not isinstance(timeframe, str):
            return False
        
        # Common timeframes - override in subclasses for specific validation
        valid_timeframes = {
            '1m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        }
        
        return timeframe.lower() in {tf.lower() for tf in valid_timeframes}
    
    def validate_date_range(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> bool:
        """Validate date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if date range is valid, False otherwise
        """
        if start_date is None and end_date is None:
            return True
        
        if start_date is not None and end_date is not None:
            if start_date >= end_date:
                return False
        
        # Check if dates are not too far in the future
        now = datetime.now()
        if start_date and start_date > now + timedelta(days=1):
            return False
        if end_date and end_date > now + timedelta(days=1):
            return False
        
        return True
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality metrics
        
        Args:
            data: OHLCV DataFrame to validate
            
        Returns:
            Dictionary with quality metrics and score
        """
        if data.empty:
            return {
                'quality_score': 0.0,
                'issues': ['Empty dataset'],
                'data_points': 0,
                'missing_ratio': 1.0
            }
        
        issues = []
        quality_score = 1.0
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 0  # O usa np.nan si prefieres marcarlo como faltante
                else:
                    raise ValueError(f"Falta la columna requerida: {col}")
        # Opcional: asegúrate de que el orden de columnas sea el correcto
        data = data[[col for col in required_columns if col in data.columns] + [c for c in data.columns if c not in required_columns]]
        
        # Check data points
        data_points = len(data)
        if data_points < self.min_data_points:
            issues.append(f"Insufficient data points: {data_points} < {self.min_data_points}")
            quality_score -= 0.2
        
        # Check missing values
        if not data.empty and len(required_columns) > 0:
            missing_count = data[required_columns].isnull().sum().sum()
            total_values = len(data) * len(required_columns)
            missing_ratio = missing_count / total_values if total_values > 0 else 1.0
            
            if missing_ratio > self.max_missing_ratio:
                issues.append(f"High missing data ratio: {missing_ratio:.2%}")
                quality_score -= 0.3
        else:
            missing_ratio = 1.0
        
        # Check for logical inconsistencies (high < low, etc.)
        if not data.empty and all(col in data.columns for col in ['high', 'low', 'open', 'close']):
            invalid_high_low = (data['high'] < data['low']).sum()
            invalid_prices = (
                (data['open'] < 0) | (data['high'] < 0) | 
                (data['low'] < 0) | (data['close'] < 0)
            ).sum()
            
            if invalid_high_low > 0:
                issues.append(f"Invalid high/low values: {invalid_high_low}")
                quality_score -= 0.2
            
            if invalid_prices > 0:
                issues.append(f"Negative prices: {invalid_prices}")
                quality_score -= 0.1
        
        # Ensure quality score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'data_points': data_points,
            'missing_ratio': missing_ratio if 'missing_ratio' in locals() else 0.0
        }
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.parquet"
    
    def load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached DataFrame or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_path = self.get_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                data = pd.read_parquet(cache_path)
                self.cache_hit_count += 1
                self.logger.debug(f"Cache hit: {cache_key}")
                return data
        except Exception as e:
            self.logger.warning(f"Cache read error for {cache_key}: {e}")
        
        return None

    def save_to_cache(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        Guarda datos en caché parquet por símbolo y año, evitando duplicados.
        """
        cache = get_global_cache()
        cache.save(symbol, timeframe, data)

    def rate_limit_check(self) -> None:
        """Check and enforce rate limiting"""
        if self.rate_limit <= 0:
            return
        
        if self.last_request_time is not None:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit:
                import time
                time.sleep(self.rate_limit - elapsed)
        
        self.last_request_time = datetime.now()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and statistics
        
        Returns:
            Dictionary with provider information
        """
        return {
            'provider_id': self.provider_id,
            'name': self.name,
            'is_connected': self.is_connected,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'cache_hit_count': self.cache_hit_count,
            'cache_hit_ratio': (
                self.cache_hit_count / max(1, self.request_count)
                if self.request_count > 0 else 0.0
            ),
            'config': self.config
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test provider connection and basic functionality
        
        Returns:
            Dictionary with test results
        """
        results = {
            'connection': False,
            'symbols_available': False,
            'data_retrieval': False,
            'errors': []
        }
        
        try:
            # Test connection
            if self.connect():
                results['connection'] = True
                
                # Test symbol retrieval
                try:
                    symbols = self.get_available_symbols()
                    if symbols and len(symbols) > 0:
                        results['symbols_available'] = True
                        
                        # Test data retrieval with first symbol
                        test_symbol = symbols[0]
                        test_data = self.get_data(
                            symbol=test_symbol,
                            timeframe='1d',
                            start_date=datetime.now() - timedelta(days=30)
                        )
                        
                        if test_data and not test_data.data.empty:
                            results['data_retrieval'] = True
                            
                except Exception as e:
                    results['errors'].append(f"Data test failed: {e}")
                    
        except Exception as e:
            results['errors'].append(f"Connection test failed: {e}")
        
        finally:
            self.disconnect()
        
        return results
    
    def __str__(self) -> str:
        """String representation of the provider"""
        status = "Connected" if self.is_connected else "Disconnected"
        return f"{self.name} ({status})"
    
    def __repr__(self) -> str:
        """Detailed representation of the provider"""
        return f"{self.__class__.__name__}(provider_id='{self.provider_id}', connected={self.is_connected})"
    
    def ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    raise ValueError(f"Falta la columna requerida: {col}")
        return df