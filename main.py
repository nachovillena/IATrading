from src.interfaces.cli_interface import CLIInterface
from src.interfaces.menu_interface import MenuInterface
import sys

def main():
    """Punto de entrada principal de la aplicación"""
    if len(sys.argv) > 1:
        # Modo CLI
        cli = CLIInterface()
        return cli.run()
    else:
        # Modo Menú Interactivo
        menu = MenuInterface()
        return menu.display_main_menu()

if __name__ == "__main__":
    sys.exit(main())