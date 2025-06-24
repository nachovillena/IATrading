import click
from config import Config
from optimize import optimize_strategy
from validate_params import validate_parameters
from train import train_models
from generate_signals import generate_signals
from tune_risk import tune_risk_confidence
from grid_search_sharpe import grid_search_sharpe
from evaluate import evaluate_performance
from utils import cargar_archivos_zip_y_csv

"""
CLI para orquestar el flujo completo del sistema de trading.
Permite optimizar estrategias, validar parámetros, entrenar modelos,
generar señales, ajustar riesgos, realizar búsqueda fina y evaluar resultados.
"""

@click.group()
def cli():
    """Grupo principal de comandos CLI."""
    pass

@cli.command()
@click.argument('strategy')
@click.option('--symbol', required=True, help='Símbolo para cargar datos (ej: EURUSD)')
def optimize(strategy: str, symbol: str):
    """Optimiza parámetros para STRATEGY con los datos de un símbolo"""
    cfg = Config.load('config/config.yaml')
    optimize_strategy(strategy, cfg, symbol)
    click.echo(f"Optimización completada para {strategy}")

# Otros comandos CLI omitidos por espacio (validar_parametros, entrenar, etc.)

@cli.command()
@click.option('--symbol', required=True, help='Símbolo a procesar (ej: EURUSD)')
@click.option('--tf', default=None, help='Temporalidades a generar (ej: M5,M15,H1)')
@click.option('--overwrite', is_flag=True, help='Sobrescribir caché existente si está activado')
def importar_zip(symbol, tf, overwrite):
    """Importa ZIPs/CSVs, resamplea y cachea por símbolo/año/timeframe"""
    tf_list = tf.split(',') if tf else None
    cargar_archivos_zip_y_csv(symbol, tf_list=tf_list, overwrite=overwrite)

@cli.command()
def validar_parametros():
    """
    Validar y filtrar parámetros desde param_grid.

    Ejemplo:
        python src/cli.py validar_parametros
    """
    validate_parameters()
    click.echo("Parámetros validados y config_master.json actualizado.")

@cli.command()
def entrenar():
    """
    Entrenar modelos a partir de config_master.json.

    Ejemplo:
        python src/cli.py entrenar
    """
    train_models()
    click.echo("Modelos entrenados y guardados.")

@cli.command()
@click.option('--periodo', default='last', help='Periodo: YYYY-MM-DD:YYYY-MM-DD o "last"')
def generar_senales(periodo: str):
    """
    Generar señales usando modelos entrenados.

    Ejemplo:
        python src/cli.py generar_senales --periodo 2024-01-01:2024-02-01
    """
    generate_signals(periodo)
    click.echo(f"Señales generadas para periodo {periodo}.")

@cli.command()
def tune_risk_confianza():
    """
    Ajustar riesgo USD y confianza para mejorar Sharpe.

    Ejemplo:
        python src/cli.py tune_risk_confianza
    """
    tune_risk_confidence()
    click.echo("Ajuste de riesgo y confianza completado.")

@cli.command()
def grid_search_sharpe():
    """
    Realizar búsqueda fina sobre SL, TP y horizonte.

    Ejemplo:
        python src/cli.py grid_search_sharpe
    """
    grid_search_sharpe()
    click.echo("Grid search fino completado.")

@cli.command()
def evaluar_rentabilidad():
    """
    Evaluar rentabilidad final: equity curves y métricas.

    Ejemplo:
        python src/cli.py evaluar_rentabilidad
    """
    evaluate_performance()
    click.echo("Evaluación de rendimiento completada.")

if __name__ == '__main__':
    cli()
