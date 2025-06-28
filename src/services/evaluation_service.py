"""Evaluation service for strategy performance assessment"""

from typing import Dict, Any
import pandas as pd
from datetime import datetime

from ..core.types import MetricsData, TradingData
from .config_service import ConfigService
from ..strategies.manager import StrategyManager
from ..data.pipeline import DataPipeline

class EvaluationService:
    """Service for strategy evaluation"""

    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.strategy_manager = StrategyManager()
        self.data_pipeline = DataPipeline(config_service)

    def evaluate_strategy(self, strategy: str, symbol: str) -> MetricsData:
        """Evaluate strategy performance"""
        try:
            # 1. Cargar configuración de la estrategia
            strategy_config = self.config_service.get_strategy_config(strategy)
            context_timeframes = strategy_config.get("context_timeframes", ["H1"])

            # 2. Cargar datos históricos para todas las temporalidades requeridas
            context_data = {}
            for tf in context_timeframes:
                trading_data = self.data_pipeline.fetch_data(symbol=symbol, timeframe=tf)
                if trading_data is None or trading_data.data.empty:
                    raise ValueError(f"No data for {symbol} [{tf}]")
                context_data[tf] = trading_data.data

            # 3. Instanciar la estrategia
            strategy_class = self.strategy_manager.get_strategy_class(strategy)
            strat_instance = strategy_class(strategy_config.get("default_params", {}))

            # 4. Generar señales usando la temporalidad principal y el contexto
            main_tf = context_timeframes[0]
            main_data = TradingData(context_data[main_tf])
            signals = strat_instance.generate_signals(main_data, context_data=context_data)

            # 5. Ejecutar backtest
            bt_result = strat_instance.backtest_simple(
                main_data=main_data.data,
                signals=signals.signals,
                context_data=context_data,
                commission=strategy_config.get("default_params", {}).get("commission", 0.0),
                slippage=strategy_config.get("default_params", {}).get("slippage", 0.0),
                stop_loss=strategy_config.get("default_params", {}).get("stop_loss", None),
                take_profit=strategy_config.get("default_params", {}).get("take_profit", None)
            )

            # 6. Calcular métricas
            metrics = {
                'sharpe_ratio': bt_result.get('sharpe', 0.0),
                'max_drawdown': bt_result.get('max_drawdown', 0.0),
                'win_rate': bt_result.get('winrate', 0.0),
                'profit_factor': bt_result.get('profit_factor', 0.0),
                'total_trades': bt_result.get('n_trades', 0),
                'total_return': bt_result.get('total_return', 0.0),
                'final_equity': bt_result.get('final_equity', 0.0)
            }

            return MetricsData(
                metrics=metrics,
                timestamp=datetime.now(),
                metadata={
                    'strategy': strategy,
                    'symbol': symbol,
                    'main_timeframe': main_tf,
                    'context_timeframes': context_timeframes
                }
            )

        except Exception as e:
            raise RuntimeError(f"Evaluation failed for {strategy}: {str(e)}")
