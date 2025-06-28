"""Custom exceptions for the trading system"""


class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    pass


class DataError(TradingSystemError):
    """Exception raised for data-related errors"""
    pass


class StrategyError(TradingSystemError):
    """Exception raised for strategy-related errors"""
    pass


class ValidationError(TradingSystemError):
    """Exception raised for validation errors"""
    pass


class ConfigurationError(TradingSystemError):
    """Exception raised for configuration errors"""
    pass


class ProviderError(TradingSystemError):
    """Exception raised for data provider errors"""
    pass


# Alias para compatibilidad
class DataProviderError(ProviderError):
    """Exception raised for data provider errors (alias for ProviderError)"""
    pass


class ExecutionError(TradingSystemError):
    """Exception raised for trade execution errors"""
    pass


class BacktestError(TradingSystemError):
    """Exception raised for backtesting errors"""
    pass


class OptimizationError(TradingSystemError):
    """Exception raised for optimization errors"""
    pass


class RiskManagementError(TradingSystemError):
    """Exception raised for risk management errors"""
    pass


# Excepciones comunes
class ConnectionError(DataError):
    """Exception raised for connection errors"""
    pass


class ApiError(DataError):
    """Exception raised for api errors"""
    pass


class TimeoutError(TradingSystemError):
    """Exception raised for timeout errors"""
    pass


class AuthenticationError(TradingSystemError):
    """Exception raised for authentication errors"""
    pass


class RateLimitError(TradingSystemError):
    """Exception raised for ratelimit errors"""
    pass


class InvalidSymbolError(TradingSystemError):
    """Exception raised for invalidsymbol errors"""
    pass


class DataQualityError(TradingSystemError):
    """Exception raised for dataquality errors"""
    pass


class CacheError(TradingSystemError):
    """Exception raised for cache errors"""
    pass

class ProcessingError(TradingSystemError):
    """Exception raised for processing errors"""
    pass

class MLModelError(TradingSystemError):
    """Exception raised for machine learning model errors"""
    pass


