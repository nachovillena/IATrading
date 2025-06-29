import pandas as pd
import matplotlib.pyplot as plt

def chequear_datos(data: dict):
    """
    Chequeo rápido de calidad de datos para cada timeframe.
    """
    for tf, df in data.items():
        print(f"--- Timeframe: {tf} ---")
        print(f"  Registros: {len(df)}")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"  Fechas: {df.index.min()} -> {df.index.max()}")
        else:
            print("  Índice no es DatetimeIndex")
        print(f"  Duplicados en índice: {df.index.has_duplicates}")
        print(f"  NaN por columna:\n{df.isna().sum()}")
        if 'close' in df.columns:
            print(f"  Precio close: min={df['close'].min()}, max={df['close'].max()}")
        print()

def graficar_datos(data: dict, mostrar_ema: bool = False, ema_fast: int = 12, ema_slow: int = 26):
    """
    Muestra un gráfico de precios para cada timeframe, opcionalmente con EMAs.
    """
    for tf, df in data.items():
        plt.figure(figsize=(12, 4))
        if 'close' in df.columns:
            plt.plot(df.index, df['close'], label='Close')
            if mostrar_ema:
                ema_f = df['close'].ewm(span=ema_fast).mean()
                ema_s = df['close'].ewm(span=ema_slow).mean()
                plt.plot(df.index, ema_f, label=f'EMA{ema_fast}')
                plt.plot(df.index, ema_s, label=f'EMA{ema_slow}')
            plt.title(f"Timeframe: {tf} - {len(df)} registros")
            plt.xlabel("Fecha")
            plt.ylabel("Precio de cierre")
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"El dataframe de {tf} no tiene columna 'close'.")

def chequear_alineacion_temporal(data: dict):
    """
    Imprime el rango de fechas de cada timeframe para comprobar alineación.
    """
    print("\n=== Alineación temporal entre timeframes ===")
    for tf, df in data.items():
        if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            print(f"{tf}: {df.index.min()} -> {df.index.max()} ({len(df)} registros)")
        else:
            print(f"{tf}: Índice no es de fechas")