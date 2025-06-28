"""Command Line Interface for the Trading System"""

import argparse

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from ..services.orchestrator import TradingOrchestrator
from ..core.exceptions import TradingSystemError
from ..utils.logger import Logger

class CLIInterface:
    """Command Line Interface for trading operations"""
    
    def __init__(self):
        """Initialize CLI interface"""
        self.orchestrator = TradingOrchestrator()
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
    
    def run(self, args: Optional[List[str]] = None) -> None:
        """Run CLI with provided arguments"""
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            if parsed_args.command == 'pipeline':
                self._run_pipeline_command(parsed_args)
            elif parsed_args.command == 'status':
                self._run_status_command()
            elif parsed_args.command == 'analyze':
                self._run_analyze_command(parsed_args)
            else:
                parser.print_help()
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.logger.error(f"CLI error: {e}", exc_info=True)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(description='AI Trading System CLI')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Run trading pipeline')
        pipeline_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., EURUSD)')
        pipeline_parser.add_argument('--timeframe', default='H1', help='Timeframe (M1, M5, M15, M30, H1, H4, D1)')
        pipeline_parser.add_argument('--strategies', help='Comma-separated strategies (ema,rsi,macd)')
        pipeline_parser.add_argument('--provider', default='auto', help='Data provider (auto, yahoo, mt5)')
        pipeline_parser.add_argument('--force-update', action='store_true', help='Force fresh data fetch')
        pipeline_parser.add_argument('--period-days', type=int, default=365, help='Data period in days')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze strategy conflicts')
        analyze_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., EURUSD)')
        analyze_parser.add_argument('--timeframe', default='H1', help='Timeframe')
        analyze_parser.add_argument('--strategies', default='ema,rsi,macd', help='Strategies to analyze')
        
        # Status command
        subparsers.add_parser('status', help='Show system status')
        
        return parser
    
    def _run_pipeline_command(self, args) -> None:
        """Run pipeline command"""
        # Parse strategies
        strategies = []
        if args.strategies:
            strategies = [s.strip() for s in args.strategies.split(',')]
        
        # Display timeframe info
        self._display_timeframe_info(args.timeframe)
        
        print(f"🚀 Running full pipeline for {args.symbol} ({args.timeframe})")
        
        # Run pipeline
        results = self.orchestrator.run_full_pipeline(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategies=strategies,
            force_update=args.force_update,
            provider=args.provider,
            period_days=args.period_days
        )
        
        # Display results
        self._display_results(results)
        
        # Show conflict analysis if multiple strategies
        if len(strategies) > 1:
            self._analyze_signal_conflicts(results)
    
    def _run_analyze_command(self, args) -> None:
        """Run analysis command"""
        strategies = [s.strip() for s in args.strategies.split(',')]
        
        print(f"🔍 Analyzing signal conflicts for {args.symbol} ({args.timeframe})")
        
        # Run pipeline to get signals
        results = self.orchestrator.run_full_pipeline(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategies=strategies
        )
        
        # Focus on conflict analysis
        self._analyze_signal_conflicts(results, detailed=True)
    
    def _run_status_command(self) -> None:
        """Run status command"""
        print("🔍 System Status:")
        status = self.orchestrator.get_system_status()
        
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Timestamp: {status.get('timestamp', 'unknown')}")
        
        # Components
        components = status.get('components', {})
        for component, status_val in components.items():
            print(f"  {component}: {status_val}")
        
        # Strategies
        strategies = status.get('strategies', {})
        print(f"Available strategies: {strategies.get('available', [])}")
        
        # Cache
        cache = status.get('cache', {})
        if cache:
            print(f"Cache: {cache.get('total_files', 0)} files, {cache.get('hit_rate', 0)}% hit rate")
    
    def _display_timeframe_info(self, timeframe: str) -> None:
        """Display timeframe availability info"""
        timeframe_info = {
            'M1': 'Up to 7 days available',
            'M5': 'Up to 60 days available', 
            'M15': 'Up to 60 days available',  # ✅ Update this
            'M30': 'Up to 60 days available',
            'H1': 'Up to 2 years available',
            'H4': 'Up to 2 years available',
            'D1': 'Up to 10 years available'
        }
        
        info = timeframe_info.get(timeframe, 'Availability unknown')
        print(f"ℹ️  {timeframe} data: {info}")
    
    def _display_results(self, results: dict) -> None:
        """Display pipeline results"""
        print(f"\n🎯 Pipeline Results for {results['symbol']} ({results['timeframe']}):")
        
        # Data status
        data_status = results.get('data_status', {})
        if data_status.get('status') == 'success':
            print(f"✅ Data Pipeline: Success - {data_status['records']:,} records")
            print(f"   📊 Quality Score: {data_status['quality_score']:.2f}")
            print(f"   🔧 Provider: {data_status['provider']}")
            if 'date_range' in data_status:
                print(f"   📅 Date Range: {data_status['date_range']['start']} to {data_status['date_range']['end']}")
        else:
            print(f"❌ Data Pipeline: Failed")
        
        # Features status
        features = results.get('features', {})
        if features.get('status') == 'success':
            feature_count = features.get('feature_count', 0)
            records = features.get('records', 0)
            print(f"✅ Feature Engineering: Success - {feature_count} features created")
            print(f"   📊 Records with features: {records:,}")
            if features.get('feature_names'):
                feature_names = ', '.join(features['feature_names'][:5])
                print(f"   🏷️  Sample features: {feature_names}...")
        else:
            print(f"❌ Feature Engineering: Failed")
            if features.get('error'):
                print(f"   Error: {features['error']}")
        
        # Strategy results
        signals = results.get('signals', {})
        if signals:
            print(f"\n📈 Strategy Results:")
            for strategy, signal_data in signals.items():
                if signal_data.get('status') == 'success':
                    signal_count = signal_data.get('signal_count', 0)
                    non_zero = signal_data.get('non_zero_signals', 0)
                    last_signal = signal_data.get('last_signal', 0)
                    
                    signal_emoji = "🟢" if last_signal > 0 else "🔴" if last_signal < 0 else "⚪"
                    signal_text = "BUY" if last_signal > 0 else "SELL" if last_signal < 0 else "HOLD"
                    
                    print(f"   {signal_emoji} {strategy}: {non_zero:,} signals, Last: {signal_text}")
                else:
                    print(f"   ❌ {strategy}: Failed")
        
        # Summary
        summary = results.get('summary', {})
        if summary:
            print(f"\n📋 Summary:")
            if summary.get('best_strategy'):
                print(f"   🎯 Best Strategy: {summary['best_strategy']}")
            print(f"   📊 Success Rate: {summary.get('success_rate', '0/0')} ({summary.get('success_percentage', 0)}%)")
            print(f"   ⏱️  Execution Time: {summary.get('execution_time', 0)}s")
        
        # Errors
        errors = results.get('errors', [])
        if errors:
            print(f"\n⚠️  Errors ({len(errors)}):")
            for error in errors[:3]:  # Show first 3 errors
                print(f"   • {error}")
    
    def _analyze_signal_conflicts(self, results: Dict[str, Any], detailed: bool = False) -> None:
        """Analyze conflicts between strategy signals"""
        signals = results.get('signals', {})
        if len(signals) < 2:
            return
        
        print(f"\n🔍 Signal Conflict Analysis:")
        
        # Get last signals from each strategy
        strategy_signals = {}
        for strategy, signal_data in signals.items():
            if signal_data.get('status') == 'success':
                last_signal = signal_data.get('last_signal', 0)
                strategy_signals[strategy] = last_signal
        
        # Analyze consensus
        buy_count = sum(1 for signal in strategy_signals.values() if signal > 0)
        sell_count = sum(1 for signal in strategy_signals.values() if signal < 0)
        hold_count = sum(1 for signal in strategy_signals.values() if signal == 0)
        total_strategies = len(strategy_signals)
        
        print(f"   📊 Signal Distribution:")
        print(f"      🟢 BUY:  {buy_count}/{total_strategies} strategies")
        print(f"      🔴 SELL: {sell_count}/{total_strategies} strategies")
        print(f"      ⚪ HOLD: {hold_count}/{total_strategies} strategies")
        
        # Determine consensus
        if buy_count > sell_count and buy_count > hold_count:
            consensus = "🟢 BULLISH CONSENSUS"
            confidence = (buy_count / total_strategies) * 100
        elif sell_count > buy_count and sell_count > hold_count:
            consensus = "🔴 BEARISH CONSENSUS"  
            confidence = (sell_count / total_strategies) * 100
        else:
            consensus = "⚠️ MIXED SIGNALS"
            confidence = max(buy_count, sell_count, hold_count) / total_strategies * 100
        
        print(f"\n   🎯 Market Consensus: {consensus}")
        print(f"   📈 Confidence Level: {confidence:.1f}%")
        
        # Risk assessment
        if confidence >= 75:
            risk_level = "🟢 LOW RISK"
        elif confidence >= 50:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🔴 HIGH RISK"
        
        print(f"   ⚠️  Risk Level: {risk_level}")
        
        # Trading recommendation
        if detailed:
            print(f"\n   💡 Trading Recommendation:")
            if confidence >= 75:
                print(f"      ✅ Strong signal - Consider position")
            elif confidence >= 50:
                print(f"      ⚠️  Wait for clearer signals")
            else:
                print(f"      ❌ Conflicting signals - Avoid trading")
