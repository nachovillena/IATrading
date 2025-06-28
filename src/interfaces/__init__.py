"""Interfaces module for user interaction"""

from .cli_interface import CLIInterface
from .menu_interface import MenuInterface

# Importación condicional de WebInterface (requiere Flask)
try:
    from .web_interface import WebInterface
    HAS_WEB_INTERFACE = True
    __all__ = ['CLIInterface', 'MenuInterface', 'WebInterface']
except ImportError:
    HAS_WEB_INTERFACE = False
    __all__ = ['CLIInterface', 'MenuInterface']
