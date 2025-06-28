"""Portfolio and risk management"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ..core.types import SignalData
from ..core.exceptions import TradingSystemError

class PortfolioManager:
    """Manages portfolio positions and allocations"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
    
    def calculate_position_size(self, signal: SignalData, risk_percent: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        
        # Simple position sizing based on available capital and risk
        risk_amount = self.current_capital * risk_percent
        
        # Assuming 1% price movement as stop loss
        stop_loss_distance = signal.price * 0.01
        
        if stop_loss_distance > 0:
            position_size = risk_amount / stop_loss_distance
        else:
            position_size = self.current_capital * 0.1  # 10% of capital as fallback
        
        return min(position_size, self.current_capital * 0.2)  # Max 20% of capital
    
    def update_position(self, symbol: str, signal: SignalData, position_size: float):
        """Update portfolio position"""
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'size': 0,
                'avg_price': 0,
                'unrealized_pnl': 0
            }
        
        # Update position
        old_size = self.positions[symbol]['size']
        new_size = signal.signal * position_size
        
        self.positions[symbol]['size'] = new_size
        self.positions[symbol]['avg_price'] = signal.price
        
        # Record trade
        self.trade_history.append({
            'timestamp': signal.timestamp,
            'symbol': symbol,
            'signal': signal.signal,
            'price': signal.price,
            'size': position_size,
            'old_position': old_size,
            'new_position': new_size
        })

class RiskManager:
    """Manages trading risk"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'max_risk_per_trade': 0.02,  # 2% per trade
            'max_portfolio_risk': 0.10,  # 10% total portfolio
            'max_correlation': 0.7,      # Max correlation between positions
            'stop_loss_percent': 0.02,   # 2% stop loss
            'take_profit_percent': 0.04  # 4% take profit
        }
    
    def evaluate_signal_risk(self, signal: SignalData, portfolio: PortfolioManager) -> Dict[str, Any]:
        """Evaluate risk for a trading signal"""
        
        risk_assessment = {
            'approved': True,
            'risk_score': 0.0,
            'warnings': [],
            'max_position_size': 0.0
        }
        
        # Calculate maximum allowed position size
        max_risk_amount = portfolio.current_capital * self.config['max_risk_per_trade']
        stop_loss_distance = signal.price * self.config['stop_loss_percent']
        
        if stop_loss_distance > 0:
            max_position_size = max_risk_amount / stop_loss_distance
        else:
            max_position_size = portfolio.current_capital * 0.05
        
        risk_assessment['max_position_size'] = max_position_size
        
        # Check portfolio concentration
        if self._check_concentration_risk(signal.symbol, portfolio):
            risk_assessment['warnings'].append('High concentration risk')
            risk_assessment['risk_score'] += 0.3
        
        # Check signal confidence
        if signal.confidence < 0.7:
            risk_assessment['warnings'].append('Low signal confidence')
            risk_assessment['risk_score'] += 0.2
        
        # Overall risk decision
        if risk_assessment['risk_score'] > 0.5:
            risk_assessment['approved'] = False
        
        return risk_assessment
    
    def _check_concentration_risk(self, symbol: str, portfolio: PortfolioManager) -> bool:
        """Check if adding position would create concentration risk"""
        
        # Simple check: don't allow more than 30% in any single symbol
        total_exposure = sum(abs(pos['size'] * pos['avg_price']) 
                           for pos in portfolio.positions.values())
        
        if total_exposure > 0:
            symbol_exposure = portfolio.positions.get(symbol, {}).get('size', 0) * \
                            portfolio.positions.get(symbol, {}).get('avg_price', 0)
            
            concentration = abs(symbol_exposure) / total_exposure
            return concentration > 0.3
        
        return False
    
    def calculate_var(self, portfolio: PortfolioManager, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        
        if not portfolio.trade_history:
            return 0.0
        
        # Simple VaR calculation based on historical returns
        df = pd.DataFrame(portfolio.trade_history)
        
        if len(df) < 30:  # Need sufficient history
            return portfolio.current_capital * 0.05  # 5% default
        
        # Calculate daily returns
        returns = df['price'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR at specified confidence level
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)
        
        return abs(var * portfolio.current_capital)
