import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Optional

def cargar_archivos_zip_y_csv(symbol: str, tf_list: Optional[list] = None, overwrite: bool = False):
    """
    Recorre data/histdata/<symbol>/*.csv y *.zip, carga y cachea los datos.
    Opcionalmente resamplea solo las temporalidades en tf_list (por defecto todas).
    Si overwrite=True, sobrescribe caché existente.
    """
    resumen = {"procesados": 0, "cacheados": 0, "omitidos": 0, "reemplazados": 0}
    print(f"\n[INICIO] Carga de archivos para símbolo: {symbol}")
    data_dir = Path(f"data/histdata/{symbol}")
    if not data_dir.exists():
        print(f"[ERROR] Directorio no encontrado: {data_dir}")
        return

    archivos = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.zip"))
    for archivo in archivos:
        print(f"\n[PROC] {archivo.suffix.upper()} {archivo.name}")
        try:
            # Llamada CORRECTA a load_data con parámetros nombrados
            df = load_data(
                path=str(archivo),
                overwrite_cache=overwrite,
                tf_list=tf_list
            )
            resumen["procesados"] += 1
            resumen["cacheados"] += 1
            print(f"[OK] {archivo.name} → {len(df)} registros")
        except Exception as e:
            print(f"[SKIP] {archivo.name} → {e}")
            resumen["omitidos"] += 1

    print(f"\n[RESUMEN] carga {symbol}:")
    for k, v in resumen.items():
        print(f"  {k}: {v}")


def load_data(path: str,
              periodo: Optional[str] = None,
              overwrite_cache: bool = False,
              tf_list: Optional[list] = None) -> pd.DataFrame:
    """
    Carga datos OHLCV desde .parquet, .csv o .zip.
    - .parquet: lee directamente
    - .csv/.zip: asigna columnas, construye timestamp, cachea en data/cache/<SYMBOL>/<SYMBOL>_<TF>_<YEAR>.parquet
    - Resamplea a temporalidades de tf_list o todas por defecto
    - Valida calidad antes de guardar: >=10 registros, <20% nulos
    - Concatena o sobrescribe según overwrite_cache
    """
    # ----- DIAGNÓSTICO -----
    print(f"[DEBUG load_data] path={path!r}, periodo={periodo!r}, overwrite_cache={overwrite_cache!r}, tf_list={tf_list!r}")
    # -----------------------
 
    cache_root = Path("data/cache")
    file = Path(path)
    df: pd.DataFrame

    # Leer parquet directamente
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)

    # Leer CSV o ZIP
    elif path.endswith(".csv") or path.endswith(".zip"):
        # Extraer CSV si es ZIP
        if path.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name:
                    raise RuntimeError("No se encontró CSV dentro del ZIP")
                raw = z.open(csv_name)
        else:
            raw = open(path, 'rb')

        # Leer sin cabecera y asignar columnas
        df = pd.read_csv(raw, header=None)
        df.columns = ['Date Stamp', 'Time Stamp', 'open', 'high', 'low', 'close', 'volume']
        raw.close()

        # Convertir a numérico
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Construir timestamp
        df['timestamp'] = pd.to_datetime(df['Date Stamp'] + ' ' + df['Time Stamp'])
        df = df.set_index('timestamp').sort_index()

        # Filtrar por periodo
        if periodo:
            inicio, fin = periodo.split(":")
            df = df.loc[inicio:fin]

        # Eliminar duplicados de índice
        df = df[~df.index.duplicated(keep='last')]

        # Validar calidad básica
        if len(df) < 10 or df.isnull().mean().mean() > 0.2:
            raise ValueError("Datos insuficientes o con demasiados nulos")

    else:
        raise ValueError("Formato de archivo no soportado")

    # Determinar símbolo, tf base y año para nombrar caché
    # path: .../EURUSD_M1_202505.zip por ejemplo
    parts = file.stem.split('_')
    symbol = parts[2] if len(parts) >= 3 else file.stem
    tf_base = parts[3][:2] if len(parts) >= 4 else 'M1'
    year = ''.join(filter(str.isdigit, parts[3]))[:4] if len(parts) >= 4 else str(df.index.year[0])

    # Temporalidades a generar
    all_tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    tfs = tf_list if tf_list else all_tfs

    # Cache para cada tf
    for tf in tfs:
        # Resample si no es M1
        if tf != tf_base:
            rule = {"M5": "5T", "M15": "15T", "M30": "30T",
                    "H1": "1H", "H4": "4H", "D1": "1D"}[tf]
            df_tf = df.resample(rule).agg({
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:
            df_tf = df

        # Validar calidad tf
        if len(df_tf) < 10 or df_tf.isnull().mean().mean() > 0.2:
            print(f"[SKIP] {symbol}_{tf}_{year}.parquet - Calidad insuficiente")
            continue

        # Ruta de caché: data/cache/<symbol>/<symbol>_<tf>_<year>.parquet
        tf_dir = cache_root / symbol
        tf_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = tf_dir / f"{symbol}_{tf}_{year}.parquet"

        # Leer existente si no overwrite
        if parquet_path.exists() and not overwrite_cache:
            df_exist = pd.read_parquet(parquet_path)
            # Concatenar y eliminar duplicados
            df_comb = pd.concat([df_exist, df_tf]).sort_index()
            df_comb = df_comb[~df_comb.index.duplicated(keep='last')]
            # Comparar calidad
            quality_new = len(df_tf) - df_tf.isnull().sum().sum()
            quality_old = len(df_exist) - df_exist.isnull().sum().sum()
            if quality_new > quality_old:
                df_tf = df_comb
                action = "APPEND"
            else:
                print(f"[SKIP] {parquet_path.name} - Calidad existente superior")
                continue
        else:
            action = "NEW" if not parquet_path.exists() else "OVERWRITE"

        # Guardar parquet
        df_tf.to_parquet(parquet_path)
        print(f"[{action}] {parquet_path.name} → {len(df_tf)} registros")

    return df

def backtest_signals(data: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    df = data.copy()
    df['returns'] = np.random.randn(len(df)) * 0.001
    return df

def extract_features(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = data.copy()
    features = pd.DataFrame(index=df.index)
    if strategy == "ema":
        features['ema_20'] = df['close'].ewm(span=20).mean()
        features['ema_50'] = df['close'].ewm(span=50).mean()
    elif strategy == "rsi":
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()
        rs = roll_up / roll_down
        features['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    else:
        features = pd.DataFrame(np.random.randn(len(df), 10), index=df.index)
    return features.fillna(0)

def load_features_labels(strategy: str, params: dict):
    n = 1000
    X = np.random.randn(n, 10)
    y = np.random.randint(0, 2, size=n)
    return X, y

def load_equity_curve(strategy: str) -> pd.Series:
    return pd.Series(np.cumsum(np.random.randn(100) * 0.001))

def compute_performance_metrics(equity: pd.Series) -> dict:
    drawdown = (equity.cummax() - equity).max()
    ganancias = equity[equity > 0].sum()
    perdidas = abs(equity[equity < 0].sum())
    profit_factor = ganancias / perdidas if perdidas != 0 else float('inf')
    expectancy = equity.mean()
    return {
        'max_drawdown': drawdown,
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }
