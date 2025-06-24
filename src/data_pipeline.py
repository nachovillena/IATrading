from prefect import flow
import logging
import pandas as pd
import yfinance as yf
import requests
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from sqlalchemy import create_engine
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check, Index
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from config import Config
from utils import cargar_archivos_zip_y_csv 

# Configuración y logging
try:
    cfg = Config.load('config/config.yaml')
except Exception:
    raise RuntimeError('No se pudo cargar config/config.yaml')

CACHE_DIR     = Path(cfg.data_paths['cache'])
HISTDATA_BASE = Path(cfg.data_paths['histdata'])
LOG_DIR       = Path('logs')

# Asegurar directorios
for sym in cfg.symbols:
    (HISTDATA_BASE / sym).mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / 'data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataPipeline')
CACHE_TTL = cfg.parameters.get('cache_ttl_days', 1)

# Esquema de validación Pandera
schema_ohlcv = DataFrameSchema(
    columns={
        'open':   Column(float, Check.ge(0)),
        'high':   Column(float, [Check.ge(0), Check(lambda s: s >= s.shift(1).fillna(s))]),
        'low':    Column(float, Check.ge(0)),
        'close':  Column(float, Check.ge(0)),
        'volume': Column(int,   Check.ge(0)),
    },
    index=Index(pa.Timestamp),
    coerce=True
)

def detect_gaps(df: pd.DataFrame, freq: str) -> List[pd.Timestamp]:
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return list(full_idx.difference(df.index))

class DataFetcher:
    def __init__(self, api_keys: Dict[str, str], db_url: Optional[str] = None):
        self.api_keys = api_keys
        self.engine   = create_engine(db_url) if db_url else None
        self.logger   = logging.getLogger(self.__class__.__name__)

    def _cache_path(self, source: str, symbol: str, tf: str) -> Path:
        return CACHE_DIR / f"{source}_{symbol}_{tf}.parquet"

    def _is_stale(self, path: Path) -> bool:
        return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) > timedelta(days=CACHE_TTL)

    def _validate_and_report(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        validated = schema_ohlcv.validate(df, lazy=True)
        gaps = detect_gaps(validated, freq)
        if gaps:
            self.logger.warning(f"{len(gaps)} gaps detectados (primer gap: {gaps[0]}) para freq={freq}")
        if validated.index.duplicated().any():
            raise pa.errors.SchemaError("Timestamps duplicados en la serie")
        return validated

    def _cache_and_store(self, df: pd.DataFrame, symbol: str, tf: str, source: str) -> pd.DataFrame:
        path = self._cache_path(source, symbol, tf)
        df.to_parquet(path)
        if self.engine:
            df.to_sql(f"{symbol}_{tf}", self.engine, if_exists='replace', index_label='timestamp')
        return df

    def _freq_to_pd(self, tf: str) -> str:
        f = tf.lower()
        if f in ['m1','1m']: return '1T'
        if f.endswith('m'):   return f.upper().replace('M','T')
        if f.endswith('h'):   return f.upper()
        return 'D'

    def fetch_local(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        path = self._cache_path('local', symbol, tf)
        if path.exists() and not self._is_stale(path):
            df = pd.read_parquet(path)
            return self._validate_and_report(df, self._freq_to_pd(tf))
        return None

    def fetch_histdata(self, symbol: str, tf: str) -> pd.DataFrame:
        histdir = HISTDATA_BASE / symbol
        zips    = list(histdir.glob(f"HISTDATA_COM_MT_{symbol}_M1*.zip"))
        if not zips:
            raise FileNotFoundError(f"No hay ZIPs HistData para {symbol}")

        with zipfile.ZipFile(zips[0]) as z:
            csv_name = next((f for f in z.namelist()
                             if 'M1' in f.upper() and f.lower().endswith('.csv')), None)
            if not csv_name:
                raise RuntimeError("CSV M1 no encontrado en ZIP de HistData")
            m1 = pd.read_csv(
                z.open(csv_name), header=None,
                names=['date','time','open','high','low','close','volume'],
                parse_dates={'timestamp': ['date','time']}
            ).set_index('timestamp').sort_index()

        m1 = self._validate_and_report(m1, '1T')
        m1 = self._cache_and_store(m1, symbol, 'M1', 'histdata')
        if tf.upper() == 'M1':
            return m1

        rule = self._freq_to_pd(tf)
        agg  = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        df_res = m1.resample(rule).agg(agg).dropna()
        df_res = self._validate_and_report(df_res, rule)
        return self._cache_and_store(df_res, symbol, tf, 'histdata')

    @retry(retry=retry_if_exception_type(Exception),
           wait=wait_exponential(1,60), stop=stop_after_attempt(3))
    def fetch_yahoo(self, symbol: str, tf: str) -> pd.DataFrame:
        if tf.lower() != 'daily':
            raise RuntimeError("Yahoo Finance solo soporta 'daily'")
        df = yf.download(symbol, interval='1d', progress=False)
        df.index.name = 'timestamp'
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        df = self._validate_and_report(df, 'D')
        return self._cache_and_store(df, symbol, 'daily', 'yahoo')

    @retry(retry=retry_if_exception_type(Exception),
           wait=wait_exponential(1,60), stop=stop_after_attempt(3))
    def fetch_binance(self, symbol: str, tf: str) -> pd.DataFrame:
        interval = tf.lower() if tf.lower() != 'daily' else '1d'
        pair     = symbol.replace('/','')
        url      = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit=1000"
        data     = requests.get(url, timeout=30).json()
        df       = pd.DataFrame(data, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','qa_volume','trades','taker_buy_base','taker_buy_quote','ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('timestamp')[['open','high','low','close','volume']].astype(float)
        df = self._validate_and_report(df, self._freq_to_pd(tf))
        return self._cache_and_store(df, symbol, tf, 'binance')


@flow
def cargar_datos(symbol: str = 'EURUSD', tf: str = 'M15'):
    """
    Flujo principal que:
      1. Carga y cachea ZIPs/CSVs locales (M1) por símbolo.
      2. Intenta fetch local, HistData, Yahoo y Binance en ese orden.
    """
    fetcher = DataFetcher(api_keys={})

    # 1) Cargar archivos M1 locales (ZIP/CSV) -> cache data/cache/<symbol>/
    cargar_archivos_zip_y_csv(symbol)

    # 2) Intentar fuentes externas si no hay datos locales
    fuentes = ["local", "histdata", "yahoo", "binance"]
    for fuente in fuentes:
        try:
            if fuente == "local":
                df = fetcher.fetch_local(symbol, tf)
            elif fuente == "histdata":
                df = fetcher.fetch_histdata(symbol, tf)
            elif fuente == "yahoo":
                df = fetcher.fetch_yahoo(symbol, tf)
            else:  # binance
                df = fetcher.fetch_binance(symbol, tf)

            if df is not None:
                logger.info(f"[OK] Datos desde {fuente}")
                print(f"{len(df)} registros obtenidos desde {fuente}.")
                return
        except Exception as e:
            logger.warning(f"[WARN] {fuente} falló: {e}")

    logger.error("[ERROR] No se pudo cargar datos desde ninguna fuente")


if __name__ == "__main__":
    cargar_datos()
