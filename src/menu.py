#!/usr/bin/env python3
import subprocess
import sys
import os
import re
from config import Config

# Validación básica para periodo
def validar_periodo(periodo):
    patron = r'^\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}$'
    return re.match(patron, periodo)

# Carga estrategias desde configuración
def load_strategies():
    cfg = Config.load('config/config.yaml')
    return list(cfg.parameters.keys())

MENU = [
    ('Cargar datos desde ZIP/CSV', 'cargar_archivos', ['symbol']),
    ('Cargar datos', 'cargar_datos', []),
    ('Optimizar parámetros', 'optimize', ['strategy']),
    ('Validar parámetros', 'validar_parametros', []),
    ('Entrenar modelos', 'entrenar', []),
    ('Generar señales', 'generar_senales', ['periodo']),
    ('Ajustar riesgo', 'tune_risk_confianza', []),
    ('Búsqueda fina SL/TP', 'grid_search_sharpe', []),
    ('Evaluar rentabilidad', 'evaluar_rentabilidad', []),
    ('Salir', None, [])
]

def print_menu():
    print("\n===== Menú Trading IA =====")
    for i, (label, _, _) in enumerate(MENU, 1):
        print(f"{i}. {label}")
    print("============================")

def main():
    strategies = load_strategies()
    script_path = os.path.join(os.path.dirname(__file__), 'cli.py')

    while True:
        print_menu()
        choice = input("Selecciona una opción (número): ")

        if not choice.isdigit() or not (1 <= int(choice) <= len(MENU)):
            print("Opción inválida. Intenta de nuevo.")
            continue

        idx = int(choice) - 1
        label, cmd, params = MENU[idx]

        if cmd is None:
            print("Saliendo del menú. Hasta luego!")
            sys.exit(0)

        args = ['python', script_path, cmd]

        for p in params:
            if p == 'strategy':
                while True:
                    strat = input(f"  Estrategia disponible ({', '.join(strategies)}): ")
                    if strat in strategies:
                        args.append(strat)
                        break
                    else:
                        print("Estrategia no válida, inténtalo de nuevo.")

            elif p == 'periodo':
                while True:
                    periodo = input("  Periodo (YYYY-MM-DD:YYYY-MM-DD): ")
                    if validar_periodo(periodo):
                        args.extend(['--periodo', periodo])
                        break
                    else:
                        print("Formato incorrecto, intenta de nuevo.")
            elif cmd == 'cargar_archivos':
                symbol = input("  Símbolo (ej: EURUSD): ")
                args = ['python', 'src/data_pipeline.py', '--symbol', symbol]

        print(f"Ejecutando: {' '.join(args)}\n")

        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error ejecutando el comando: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupción del usuario, saliendo...")
    except Exception as e:
        print(f"Error inesperado: {e}")
