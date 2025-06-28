"""Trading module - legacy, kept for compatibility"""

# Note: Strategies have been moved to src/strategies/
# This module is kept for backward compatibility

import warnings

warnings.warn(
    "The trading.strategies module has been moved to strategies. "
    "Please update your imports to use 'from ..strategies import StrategyManager'",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for compatibility
try:
    from ..strategies import StrategyManager
    __all__ = ['StrategyManager']
except ImportError:
    __all__ = []
