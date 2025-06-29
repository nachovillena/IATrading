import pandas as pd
import numpy as np
from ...core.base.strategy import BaseStrategy
from ...core.types import TradingData, SignalData
from typing import Dict, Any, Optional

class PivotBreakoutStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] = None):      
        super().__init__(config)
        params = {**self.config.get('default_params', {}), **self.config}
        self.params = params
        self.exit_bars = params.get('exit_bars', 5)
        self.apply_sma_filter = bool(params.get('apply_sma_filter', True))
        self.sma_period = params.get('sma_period', 14)
        self.apply_roc_filter = bool(params.get('apply_roc_filter', False))
        self.roc_period = params.get('roc_period', 120)

    def get_default_parameters(self):
        return {
            "exit_bars": 5,
            "apply_sma_filter": True,
            "sma_period": 14,
            "apply_roc_filter": False,
            "roc_period": 120,
        }

    @staticmethod
    def calculate_pivots(df: pd.DataFrame) -> pd.DataFrame:
        pivots = []
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            P = (prev['high'] + prev['low'] + prev['close']) / 3
            R1 = 2 * P - prev['low']
            S1 = 2 * P - prev['high']
            R2 = P + (prev['high'] - prev['low'])
            S2 = P - (prev['high'] - prev['low'])
            pivots.append({'P': P, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2})
        pivots_df = pd.DataFrame(pivots, index=df.index[1:])
        return pivots_df

    def calculate_indicators(self, data: TradingData) -> pd.DataFrame:
        df = data.data.copy()
        df['SMA'] = df['close'].rolling(self.sma_period).mean()
        df['ROC'] = df['close'].pct_change(self.roc_period)
        return df

    def generate_signals(self, data: TradingData, context_data: Optional[Dict] = None) -> SignalData:
        df = data.data.copy()
        pivots = self.calculate_pivots(df)
        df = df.iloc[1:]  # Alinear con pivots
        df = df.join(pivots)
        df['signal'] = 0

        # ATR para filtro de rango mínimo
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()

        # Filtro SMA (tendencia clara)
        if self.apply_sma_filter:
            df['SMA'] = df['close'].rolling(self.sma_period).mean()
            df = df[df['SMA'] > df['SMA'].shift(3)]

        # Filtro ROC (momentum real)
        if self.apply_roc_filter:
            df['ROC'] = df['close'].pct_change(self.roc_period)
            df = df[df['ROC'] > 0.002]  # Ajusta el umbral según tu activo

        # Filtro rango mínimo de vela (más estricto)
        min_range = df['ATR'].mean() * 0.8
        df = df[df['high'] - df['low'] > min_range]

        # Señal: rompe R2 en la vela
        df.loc[df['high'] >= df['R2'], 'signal'] = 1

        # Evitar operar viernes
        df = df[df.index.dayofweek < 4]

        # Solo una señal por día
        df['date'] = df.index.date
        df = df[df['signal'] == 1].groupby('date').head(1)

        # Reconstruir la serie de señales para todo el histórico
        signals = pd.Series(0, index=data.data.index)
        signals.loc[df.index] = 1

        signal_dates = df
        print("Total señales:", len(signal_dates))
        return SignalData(signals=signals, metadata={}, strategy_name=self.strategy_name)

    def backtest_simple(
        self,
        main_data: TradingData,
        signals: Optional[pd.Series] = None,
        context_data: Optional[Dict] = None,
        initial_balance: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        atr_period: int = 14,
        atr_mult_sl: float = 2.0,
        atr_mult_tp: float = 4.0
    ) -> dict:
        self.validate_data(main_data)
        df = main_data.data.copy()
        if signals is None:
            signals = self.generate_signals(main_data, context_data=context_data).signals

        df['signal'] = signals
        df['returns'] = df['close'].pct_change().fillna(0)

        # ATR para stops dinámicos
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(atr_period).mean()

        exit_bars = self.exit_bars
        equity = initial_balance
        trade_cost = commission + slippage
        num_trades = 0

        # Para métricas adicionales
        equity_curve = [equity]
        trade_profits = []
        wins = 0
        losses = 0
        gross_profit = 0
        gross_loss = 0

        # NUEVA LÓGICA: Solo una operación abierta a la vez
        in_trade = False
        last_exit_idx = None

        for idx, row in df.iterrows():
            # Solo abrir trade si hay señal y no hay operación abierta
            if row['signal'] == 1 and not in_trade and not pd.isna(row['ATR']):
                entry_price = row['close']
                atr = row['ATR']
                sl = entry_price - atr * atr_mult_sl
                tp = entry_price + atr * atr_mult_tp
                equity -= equity * trade_cost
                num_trades += 1
                bars_in_trade = 0
                in_trade = True

                # Busca la salida (stop, take profit o exit_bars)
                for j in range(1, exit_bars + 1):
                    # Busca el índice siguiente
                    if idx not in df.index:
                        break
                    next_idx = df.index.get_loc(idx) + j
                    if next_idx >= len(df):
                        break
                    next_row = df.iloc[next_idx]
                    bars_in_trade += 1
                    # Stop-loss
                    if next_row['low'] <= sl:
                        ret = (sl - entry_price) / entry_price
                        profit = equity * ret
                        equity *= (1 + ret)
                        equity -= equity * trade_cost
                        trade_profits.append(profit)
                        if profit > 0:
                            wins += 1
                            gross_profit += profit
                        else:
                            losses += 1
                            gross_loss += abs(profit)
                        equity_curve.append(equity)
                        in_trade = False
                        last_exit_idx = df.index[next_idx]
                        break
                    # Take-profit
                    if next_row['high'] >= tp:
                        ret = (tp - entry_price) / entry_price
                        profit = equity * ret
                        equity *= (1 + ret)
                        equity -= equity * trade_cost
                        trade_profits.append(profit)
                        if profit > 0:
                            wins += 1
                            gross_profit += profit
                        else:
                            losses += 1
                            gross_loss += abs(profit)
                        equity_curve.append(equity)
                        in_trade = False
                        last_exit_idx = df.index[next_idx]
                        break
                # Si no salió por stop ni take, salir por exit_bars
                if in_trade:
                    exit_idx = df.index.get_loc(idx) + bars_in_trade
                    if exit_idx < len(df):
                        exit_row = df.iloc[exit_idx]
                        ret = (exit_row['close'] - entry_price) / entry_price
                        profit = equity * ret
                        equity *= (1 + ret)
                        equity -= equity * trade_cost
                        trade_profits.append(profit)
                        if profit > 0:
                            wins += 1
                            gross_profit += profit
                        else:
                            losses += 1
                            gross_loss += abs(profit)
                        equity_curve.append(equity)
                    in_trade = False
                    last_exit_idx = df.index[exit_idx] if exit_idx < len(df) else None

        # Métricas adicionales
        total_return = (equity / initial_balance) - 1
        max_drawdown = 0
        peak = equity_curve[0]
        for x in equity_curve:
            if x > peak:
                peak = x
            dd = (peak - x) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        win_rate = (wins / num_trades) if num_trades > 0 else 0

        results = {
            "final_equity": equity,
            "num_trades": num_trades,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "trade_profits": trade_profits,
            "equity_curve": equity_curve,
        }
        return results