"""Trading orchestrator service"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.types import TradingData
from ..core.exceptions import TradingSystemError
from ..data.pipeline import DataPipeline
from ..strategies import StrategyManager
from ..ml.features import FeatureEngineer
from ..utils.logger import Logger

class TradingOrchestrator:
    """Orchestrates the complete trading pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading orchestrator"""
        self.config = config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name

        # Initialize components
        self.data_pipeline = DataPipeline(self.config.get('data_pipeline'))
        self.strategy_manager = StrategyManager(self.config.get('strategies'))
        self.feature_engineer = FeatureEngineer(self.config.get('features'))

        self.logger.info("Trading Orchestrator initialized")

    def run_full_pipeline(self, symbol: str, timeframe: str = 'H1',
                         strategies: Optional[List[str]] = None,
                         force_update: bool = False,
                         provider: str = 'auto',
                         period_days: int = 365,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Run the complete trading pipeline for a symbol"""
        results = self._initialize_results(symbol, timeframe)

        try:
            # Step 1: Data acquisition
            data = self._fetch_data_step(symbol, timeframe, force_update,
                                         provider, period_days, start_date, end_date, results)

            # Step 2: Feature engineering
            features = self._feature_engineering_step(data, results)

            # Step 3: Strategy processing
            if strategies:
                self._strategy_processing_step(data, strategies, timeframe, results)

            # Step 4: Generate summary
            self._generate_summary_step(strategies, results)

        except Exception as e:
            self._handle_pipeline_error(e, results)

        return results

    def _initialize_results(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Initialize results dictionary"""
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'data_status': {},
            'features': {},
            'signals': {},
            'errors': [],
            'summary': {}
        }

    def _fetch_data_step(self, symbol: str, timeframe: str, force_update: bool,
                        provider: str, period_days: int,
                        start_date: Optional[datetime], end_date: Optional[datetime],
                        results: Dict[str, Any]) -> TradingData:
        """Step 1: Data acquisition and quality control"""
        self.logger.info(f"Step 1: Data acquisition and quality control ({timeframe})")

        # Usar el nuevo sistema: si force_update=True, descarga y guarda; si False, solo caché
        data = self.data_pipeline.fetch_data(
            symbol=symbol,
            timeframe=timeframe,
            provider=provider,
            force_update=force_update,
            allow_download=force_update,  # Si force_update, permitimos descarga
            start_date=start_date,
            end_date=end_date
        )

        # Check if data is valid
        if data is None or data.data.empty:
            raise TradingSystemError(f"No data available for {symbol} ({timeframe})")

        results['data_status'] = {
            'status': 'success',
            'records': len(data.data),
            'timeframe': timeframe,
            'provider': provider,
            'date_range': {
                'start': data.data.index[0].isoformat(),
                'end': data.data.index[-1].isoformat()
            },
            'quality_score': getattr(data, 'quality_score', 0.8)
        }

        return data

    def _feature_engineering_step(self, data: TradingData, results: Dict[str, Any]) -> pd.DataFrame:
        """Step 2: Feature engineering"""
        self.logger.info("Step 2: Feature engineering")

        try:
            # Create features
            features_df = self.feature_engineer.create_features(data)

            # Identify original vs new columns
            original_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
            all_cols = list(features_df.columns)
            feature_cols = [col for col in all_cols if col not in original_cols]

            results['features'] = {
                'status': 'success',
                'feature_count': len(feature_cols),
                'records': len(features_df),
                'total_columns': len(all_cols),
                'original_columns': len(original_cols),
                'feature_names': feature_cols[:10] if feature_cols else []
            }

            self.logger.info(f"✅ Created {len(feature_cols)} features for {len(features_df)} records")
            return features_df

        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}", exc_info=True)
            results['features'] = {
                'status': 'failed',
                'error': str(e),
                'feature_count': 0,
                'records': 0
            }
            results['errors'].append(f"Feature engineering failed: {e}")
            return pd.DataFrame()

    def _strategy_processing_step(self, data: TradingData, strategies: List[str],
                                 timeframe: str, results: Dict[str, Any]) -> None:
        """Step 3: Strategy processing"""
        self.logger.info("Step 3: Strategy processing")

        try:
            # Process signals for all requested strategies
            signals_results = self.strategy_manager.process_signals(data, strategies)

            # Store results for each strategy
            for strategy_name, signal_data in signals_results.items():
                if hasattr(signal_data, 'signals') and hasattr(signal_data.signals, '__len__'):
                    signal_count = len(signal_data.signals)

                    # Count actual non-zero signals
                    if hasattr(signal_data.signals, 'values'):
                        non_zero_signals = int((signal_data.signals != 0).sum())
                        buy_signals = int((signal_data.signals > 0).sum())
                        sell_signals = int((signal_data.signals < 0).sum())
                    else:
                        non_zero_signals = 0
                        buy_signals = 0
                        sell_signals = 0

                    last_signal = signal_data.metadata.get('last_signal', 0) if hasattr(signal_data, 'metadata') else 0
                else:
                    signal_count = 0
                    non_zero_signals = 0
                    buy_signals = 0
                    sell_signals = 0
                    last_signal = 0

                results['signals'][strategy_name] = {
                    'status': 'success',
                    'signal_count': signal_count,
                    'non_zero_signals': non_zero_signals,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'last_signal': last_signal,
                    'metadata': signal_data.metadata if hasattr(signal_data, 'metadata') else {}
                }

                self.logger.info(f"Strategy {strategy_name}: {signal_count} total, {non_zero_signals} actual signals")

        except Exception as e:
            self.logger.error(f"Strategy processing failed: {e}")
            results['errors'].append(f"Strategy processing failed: {e}")

    def _generate_summary_step(self, strategies: Optional[List[str]], results: Dict[str, Any]) -> None:
        """Step 4: Generate summary"""
        self.logger.info("Step 4: Generating summary")

        try:
            # Calculate execution time
            execution_time = (datetime.now() - results['timestamp']).total_seconds()

            # Find best performing strategy (one with most signals)
            best_strategy = None
            max_signals = 0

            if strategies and results.get('signals'):
                for strategy in strategies:
                    strategy_data = results['signals'].get(strategy, {})
                    if strategy_data.get('status') == 'success':
                        signal_count = strategy_data.get('non_zero_signals', 0)
                        if signal_count > max_signals:
                            max_signals = signal_count
                            best_strategy = strategy
                        if best_strategy is None:
                            best_strategy = strategy

            # Count successful steps correctly
            successful_steps = 0
            total_steps = 0

            # Data step
            if results.get('data_status', {}).get('status') == 'success':
                successful_steps += 1
            total_steps += 1

            # Features step
            if results.get('features', {}).get('status') == 'success':
                successful_steps += 1
            total_steps += 1

            # Strategy steps
            if strategies:
                strategy_successes = sum(1 for s in strategies if results.get('signals', {}).get(s, {}).get('status') == 'success')
                successful_steps += strategy_successes
                total_steps += len(strategies)

            results['summary'] = {
                'execution_time': round(execution_time, 2),
                'success_rate': f"{successful_steps}/{total_steps}",
                'success_percentage': round((successful_steps / total_steps) * 100, 1) if total_steps > 0 else 0,
                'best_strategy': best_strategy,
                'total_errors': len(results.get('errors', [])),
                'completed_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            results['summary'] = {'error': str(e)}

        self.logger.info(f"Pipeline completed for {results['symbol']}")

    def _handle_pipeline_error(self, error: Exception, results: Dict[str, Any]) -> None:
        """Handle pipeline errors"""
        error_msg = f"Pipeline failed: {str(error)}"
        self.logger.error(error_msg)
        results['errors'].append(error_msg)
        results['summary'] = {
            'status': 'failed',
            'error': str(error),
            'completed_at': datetime.now().isoformat()
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            cache_stats = self.data_pipeline.get_cache_stats()
            available_strategies = self.strategy_manager.get_available_strategies()

            return {
                'status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'data_pipeline': 'operational',
                    'strategy_manager': 'operational',
                    'feature_engineer': 'operational'
                },
                'cache': cache_stats,
                'strategies': {
                    'available': available_strategies,
                    'count': len(available_strategies)
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_optimization(self, strategy: str, symbol: str, trials: int = 100) -> dict:
        """
        Ejecuta la optimización de hiperparámetros para una estrategia y símbolo dados,
        usando el contexto de temporalidades definido en la configuración de la estrategia.
        Si alguna temporalidad no tiene datos, la omite y muestra un aviso.
        """
        self.logger.info(f"🔎 Iniciando optimización para {strategy.upper()} en {symbol} con {trials} trials")

        try:
            # 1. Obtener temporalidades requeridas desde el StrategyManager
            required_timeframes = self.strategy_manager.get_required_timeframes(strategy)
            self.logger.info(f"Temporalidades requeridas para {strategy}: {required_timeframes}")

            # --- NUEVO: Mostrar estado de la caché ---
            self.print_cache_status(symbol, required_timeframes)

            # 2. Cargar los históricos de todas las temporalidades requeridas
            data_by_timeframe = {}
            missing_timeframes = []
            for tf in required_timeframes:
                # Aquí puedes decidir si quieres permitir descarga o solo caché
                data = self.data_pipeline.fetch_data(
                    symbol=symbol,
                    timeframe=tf,
                    force_update=False,
                    allow_download=False  # Solo caché para optimización
                )
                if data is None or data.data.empty:
                    msg = f"⚠️  No hay datos disponibles para {symbol} [{tf}]. Se omite esta temporalidad."
                    print(msg)
                    self.logger.warning(msg)
                    missing_timeframes.append(tf)
                    continue
                data_by_timeframe[tf] = data.data

            if not data_by_timeframe:
                error_msg = f"No hay datos disponibles en ninguna de las temporalidades requeridas para {symbol}."
                print(error_msg)
                self.logger.error(error_msg)
                return {
                    "best_params": {},
                    "best_score": None,
                    "error": error_msg
                }

            # 3. Ejecutar optimización usando el optimizador de hiperparámetros
            optimizer = getattr(self, "optimizer", None)
            if optimizer is None:
                from ..ml.optimization import HyperparameterOptimizer
                optimizer = HyperparameterOptimizer()
                self.optimizer = optimizer

            # 4. Pasar todos los históricos al optimizador
            result = optimizer.optimize(
                strategy=strategy,
                symbol=symbol,
                data=data_by_timeframe,
                trials=trials
            )

            # Almacenar el mejor resultado con el nuevo campo "num_trades"
            best_params = result.get("best_params", {})
            best_score = result.get("best_score", None)
            best_num_trades = result.get("num_trades", 0)  # <-- Añade esto

            # Al guardar los mejores resultados tras la optimización
            best_result = {
                "best_params": best_params,
                "best_score": best_score,
                "num_trades": best_num_trades,  # <-- Añade esto
                # ...otros campos...
            }

            self.logger.info(f"✅ Optimización completada para {strategy.upper()} en {symbol}")
            return result

        except Exception as e:
            self.logger.error(f"Error en optimización de {strategy} para {symbol}: {e}")
            return {
                "best_params": {},
                "best_score": None,
                "error": str(e)
            }

    def print_cache_status(self, symbol: str, required_timeframes: list):
        self.logger.info(f"Estado de la caché para {symbol}:")
        for tf in required_timeframes:
            # Solo consulta caché, no descarga
            data = self.data_pipeline.fetch_data(
                symbol=symbol,
                timeframe=tf,
                force_update=False,
                allow_download=False
            )
            if data is None or data.data.empty:
                print(f"❌ Sin datos en caché para {tf}")
            else:
                print(f"✅ Datos en caché para {tf} ({len(data.data)} registros)")
        self.logger.info(f"Fin del estado de la caché para {symbol}")
