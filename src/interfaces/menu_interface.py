"""Interactive menu interface for the trading system"""

import os
import sys
from typing import Dict, List, Any
from pathlib import Path
import yaml
from datetime import datetime, timedelta

from ..services.orchestrator import TradingOrchestrator
from ..core.exceptions import TradingSystemError
from ..data.pipeline import DataPipeline
from ..ml.optimization import HyperparameterOptimizer
from ..trading.signals import SignalManager
from ..utils.logger import Logger
from ..utils.utils import graficar_datos

class MenuInterface:
    """Interactive menu interface for the trading system"""

    def __init__(self):
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.orchestrator = TradingOrchestrator()
        self.data_pipeline = DataPipeline()
        self.signal_manager = SignalManager()
        self.optimizer = HyperparameterOptimizer()
        self.symbols = self.load_symbols_from_config()
        self.available_strategies = self.load_strategies_from_dir()

        # Menú principal unificado
        self.menu_options = [
            ("Ejecutar flujo completo de análisis y optimización", self.run_full_workflow),
            ("Importar datos externos (API)", self.import_external_data_with_provider_with_dates),
            ("Importar histórico completo MT5", self.importar_historico_mt5_menu),
            ("Importar archivo local", self.import_local_file),
            ("Validar calidad de datos", self.validate_data_quality),
            ("Limpiar y procesar datos", self.clean_process_data),
            ("Reporte de calidad", self.quality_report),
            ("Forzar actualización de datos", self.force_data_update),
            ("Verificar estructura data/cache", self.verify_structure),
            ("Ver estado de la caché para un símbolo", self.show_cache_status_for_symbol),
            ("Optimizar estrategia específica", self.optimize_specific_strategy),
            ("Optimizar todas las estrategias", self.optimize_all_strategies),
            ("Ver resultados de optimización", self.view_optimization_results),
            ("Análisis de hiperparámetros", self.hyperparameter_analysis),
            ("Generar señales (todas las estrategias)", self.generate_all_signals),
            ("Generar señales (estrategia específica)", self.generate_specific_signals),
            ("Ver estado de señales generadas", self.view_signals_status),
            ("Exportar señales a CSV", self.export_signals),
            ("Evaluar rendimiento completo", self.evaluate_performance),
            ("Comparar estrategias", self.compare_strategies),
            ("Análisis de riesgo", self.risk_analysis),
            ("Reporte de métricas", self.metrics_report),
            ("Ver configuración actual", self.view_config),
            ("Limpiar archivos temporales", self.clean_temp_files),
            ("Ejecutar diagnósticos", self.run_diagnostics),
            ("Ver estado del sistema", self.system_status),
            ("Backup de configuración", self.backup_config),
            ("Salir", None),
            ("Visualizar datos históricos (gráfica)", self.visualizar_datos_historicos)
        ]
    def importar_historico_mt5_menu(self):
        """Importar histórico completo de MT5 para un símbolo"""
        print("📊 IMPORTAR HISTÓRICO COMPLETO MT5")
        symbol = self.get_symbol_input("Seleccione símbolo para importar histórico")
        importar_mt5_historico(self.data_pipeline, symbol)
        print("✅ Importación masiva finalizada.")

    def run_full_workflow(self):
        """Run the full workflow: import data, optimize, generate signals, backtest, analyze"""
        print("=" * 80)
        print("🚀 FLUJO COMPLETO DE ANÁLISIS Y OPTIMIZACIÓN")
        print("=" * 80)

        # 1. Selección de símbolo y estrategia(s)
        symbol = self.get_symbol_input("Seleccione símbolo para el análisis")
        strategies = self.get_strategy_input("Seleccione estrategia para el análisis")

        # 2. Importar datos (opcional en cualquier momento)
        while True:
            print("\n¿Desea importar o actualizar datos antes de continuar?")
            print("  1. Sí, importar/actualizar datos")
            print("  2. No, continuar con el flujo")
            choice = input("Opción: ").strip()
            if choice == "1":
                self.import_external_data_with_provider_with_dates()
            elif choice == "2":
                break

        # 3. Ejecutar el flujo para cada estrategia seleccionada
        for strategy in strategies:
            print(f"\n=== Procesando estrategia: {strategy.upper()} ===")
            # 3. Optimización de la estrategia
            print("\n🎯 OPTIMIZANDO ESTRATEGIA...")
            try:
                trials = input("Número de trials para optimización (default 100): ").strip()
                trials = int(trials) if trials else 100
                self.logger.info(f"🔄 Optimizando {strategy.upper()} para {symbol} con {trials} trials...")
                opt_results = self.orchestrator.run_optimization(strategy, symbol, trials)
            except Exception as e:
                self.logger.error(f"❌ Error en optimización: {e}")
                continue

            # 4. Generación de señales
            print("\n📈 GENERANDO SEÑALES CON PARÁMETROS ÓPTIMOS...")
            try:
                signals = self.signal_manager.generate_signals(strategy, symbol)
                print(f"✅ {len(signals)} señales generadas para {strategy.upper()}")
            except Exception as e:
                print(f"❌ Error al generar señales: {e}")
                continue

            # 5. Backtesting y evaluación
            print("\n📊 REALIZANDO BACKTEST Y EVALUACIÓN...")
            try:
                eval_results = self.orchestrator.run_full_evaluation(strategy, symbol)
                print("\n📈 Resultados de evaluación:")
                for k, v in eval_results.items():
                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"❌ Error en evaluación: {e}")
                continue

            # 6. Análisis de métricas y reporte
            print("\n📋 ANÁLISIS DE MÉTRICAS Y REPORTE...")
            try:
                metrics_report = self.orchestrator.generate_metrics_report(strategy, symbol)
                print("\n📊 Reporte de métricas generado:")
                print(metrics_report)
            except Exception as e:
                print(f"❌ Error al generar reporte: {e}")

        print("\n🚀 Flujo completo finalizado.")
    
    def display_main_menu(self):
        """Display the unified main menu"""
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 80)
            print("🚀 SISTEMA DE TRADING IA - MENÚ PRINCIPAL")
            print("=" * 80)
            for idx, (desc, _) in enumerate(self.menu_options, 1):
                print(f"{idx}. {desc}")
            print("=" * 80)
            try:
                choice = input("\n🔹 Seleccione una opción: ").strip()
                if not choice.isdigit() or not (1 <= int(choice) <= len(self.menu_options)):
                    print("❌ Opción no válida")
                    input("\n📌 Presione Enter para continuar...")
                    continue
                idx = int(choice) - 1
                if self.menu_options[idx][1] is None:
                    print("\n👋 ¡Hasta luego!")
                    break
                self.menu_options[idx][1]()
                input("\n📌 Presione Enter para volver al menú...")
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                input("\n📌 Presione Enter para continuar...")

    def get_symbol_input(self, prompt: str = "Seleccione símbolo") -> str:
        """Get symbol input from user"""
        print(f"\n{prompt}:")
        for i, symbol in enumerate(self.symbols, 1):
            print(f"  {i}. {symbol}")
        
        while True:
            try:
                choice = int(input("\nOpción: "))
                if 1 <= choice <= len(self.symbols):
                    return self.symbols[choice - 1]
                print("❌ Opción no válida")
            except ValueError:
                print("❌ Por favor ingrese un número")
    
    def get_strategy_input(self, prompt: str = "Seleccione estrategia") -> List[str]:
        """Get strategy input from user. Si no se indica, devuelve todas."""
        print(f"\n{prompt}:")
        for i, strategy in enumerate(self.available_strategies, 1):
            print(f"  {i}. {strategy.upper()}")
        print(f"  0. TODAS LAS ESTRATEGIAS")

        while True:
            try:
                choice = input("\nOpción (puede ser un número o varios separados por coma, 0 para todas): ").strip()
                if choice == "0" or choice == "":
                    # Si el usuario pulsa 0 o Enter, selecciona todas
                    return self.available_strategies
                # Permitir selección múltiple separada por coma
                indices = [int(x) for x in choice.split(",") if x.strip().isdigit()]
                if all(1 <= idx <= len(self.available_strategies) for idx in indices):
                    return [self.available_strategies[idx - 1] for idx in indices]
                print("❌ Opción no válida")
            except ValueError:
                print("❌ Por favor ingrese un número o varios separados por coma")

    def import_external_data_with_provider_with_dates(self):
        """Importar datos desde un proveedor externo seleccionado (Yahoo o MT5) con fechas personalizadas"""
        print("📊 IMPORTAR DATOS EXTERNOS (SELECCIÓN DE PROVIDER Y FECHAS)")
        print("-" * 30)

        # Selección de provider
        providers = ['yahoo', 'mt5']
        print("\nSeleccione proveedor de datos:")
        for i, prov in enumerate(providers, 1):
            print(f"  {i}. {prov.upper()}")
        while True:
            try:
                prov_choice = int(input("\nProveedor: "))
                if 1 <= prov_choice <= len(providers):
                    provider = providers[prov_choice - 1]
                    break
                print("❌ Opción no válida")
            except ValueError:
                print("❌ Por favor ingrese un número")

        symbol = self.get_symbol_input()

        print("\nOpciones de timeframe:")
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        for i, tf in enumerate(timeframes, 1):
            print(f"  {i}. {tf}")
        tf_choice = int(input("\nTimeframe: "))
        timeframe = timeframes[tf_choice - 1]

        # Solicitar fechas
        def ask_date(prompt):
            while True:
                date_str = input(prompt).strip()
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    print("❌ Formato inválido. Usa AAAA-MM-DD.")

        start_date = ask_date("Fecha de inicio (AAAA-MM-DD): ")
        end_date = ask_date("Fecha final (AAAA-MM-DD): ")

        try:
            print(f"\n🔄 Importando datos {symbol} {timeframe} desde {provider.upper()} entre {start_date.date()} y {end_date.date()}...")
            result = self.data_pipeline.fetch_data(
                symbol, timeframe, provider=provider,
                start_date=start_date, end_date=end_date,
                allow_download=True  # Permite descarga si no hay en caché
            )
            print(f"✅ Datos importados: {len(result.data)} registros")
        except Exception as e:
            print(f"❌ Error al importar datos: {e}")
        
    def import_local_file(self):
            """Import data from local file"""
            print("📁 IMPORTAR ARCHIVO LOCAL")
            print("-" * 30)
            
            file_path = input("Ruta del archivo: ").strip()
            
            if not Path(file_path).exists():
                print("❌ Archivo no encontrado")
                return
            
            try:
                result = self.data_pipeline.import_from_file(file_path)
                print(f"✅ Archivo importado: {len(result)} registros")
                
            except Exception as e:
                print(f"❌ Error al importar archivo: {e}")
    
    # Data Management Methods
    def import_external_data(self):
        """Import data from external API"""
        print("📊 IMPORTAR DATOS EXTERNOS")
        print("-" * 30)
        
        symbol = self.get_symbol_input()
        
        print("\nOpciones de timeframe:")
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        for i, tf in enumerate(timeframes, 1):
            print(f"  {i}. {tf}")
        
        tf_choice = int(input("\nTimeframe: "))
        timeframe = timeframes[tf_choice - 1]
        
        try:
            print(f"\n🔄 Importando datos {symbol} {timeframe}...")
            result = self.data_pipeline.fetch_data(symbol, timeframe, source="auto")
            print(f"✅ Datos importados: {len(result)} registros")
            
        except Exception as e:
            print(f"❌ Error al importar datos: {e}")

    
    def validate_data_quality(self):
        """Validate data quality"""
        print("✅ VALIDAR CALIDAD DE DATOS")
        print("-" * 30)
        
        try:
            quality_results = self.data_pipeline.quality_controller.check_all_data()
            
            print("\n📊 Resumen de calidad:")
            for symbol, results in quality_results.items():
                print(f"\n{symbol}:")
                for check, status in results.items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {check}")
            
        except Exception as e:
            print(f"❌ Error en validación: {e}")
    
    def clean_process_data(self):
        """Clean and process data"""
        print("🧹 LIMPIAR Y PROCESAR DATOS")
        print("-" * 30)
        
        symbol = self.get_symbol_input()
        
        try:
            print(f"\n🔄 Procesando datos para {symbol}...")
            result = self.data_pipeline.clean_and_process(symbol)
            print(f"✅ Datos procesados: {len(result)} registros válidos")
            
        except Exception as e:
            print(f"❌ Error al procesar datos: {e}")
    
    def quality_report(self):
        """Generate quality report"""
        print("📋 REPORTE DE CALIDAD")
        print("-" * 30)
        
        try:
            report = self.data_pipeline.quality_controller.generate_report()
            print("\n📊 Reporte de calidad generado:")
            print(report)
            
        except Exception as e:
            print(f"❌ Error al generar reporte: {e}")
    
    def force_data_update(self):
        """Force data update"""
        print("🔄 FORZAR ACTUALIZACIÓN DE DATOS")
        print("-" * 30)
        
        symbol = self.get_symbol_input()
        print("\nOpciones de timeframe:")
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        for i, tf in enumerate(timeframes, 1):
            print(f"  {i}. {tf}")
        tf_choice = int(input("\nTimeframe: "))
        timeframe = timeframes[tf_choice - 1]

        try:
            print(f"\n🔄 Actualizando datos para {symbol} [{timeframe}]...")
            result = self.data_pipeline.fetch_data(
                symbol, timeframe, force_update=True, allow_download=True
            )
            print(f"✅ Datos actualizados: {len(result.data)} registros")
        except Exception as e:
            print(f"❌ Error al actualizar datos: {e}")
    
    def verify_structure(self):
        """Verify data structure"""
        print("🔍 VERIFICAR ESTRUCTURA DATA/CACHE")
        print("-" * 30)
        
        try:
            structure_ok = self.data_pipeline.verify_data_structure()
            if structure_ok:
                print("✅ Estructura de datos correcta")
            else:
                print("❌ Problemas en la estructura de datos")
                
        except Exception as e:
            print(f"❌ Error al verificar estructura: {e}")
    
    # Optimization Methods
    def optimize_specific_strategy(self):
        """Optimize specific strategy"""
        print("🎯 OPTIMIZAR ESTRATEGIA ESPECÍFICA")
        print("-" * 30)
        
        strategy = self.get_strategy_input()
        symbol = self.get_symbol_input()
        
        trials = input("\nNúmero de trials (default 100): ").strip()
        trials = int(trials) if trials else 100
        
        try:
            print(f"\n🔄 Optimizando {strategy.upper()} para {symbol}...")
            results = self.orchestrator.run_optimization(strategy, symbol, trials)
            print(f"✅ Optimización completada")
            print(f"📊 Mejores parámetros: {results.get('best_params', {})}")
            print(f"🎯 Mejor score: {results.get('best_score', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error en optimización: {e}")
    
    def optimize_all_strategies(self):
        """Optimize all strategies"""
        print("🎯 OPTIMIZAR TODAS LAS ESTRATEGIAS")
        print("-" * 30)
        
        symbol = self.get_symbol_input()
        
        try:
            for strategy in self.available_strategies:
                print(f"\n🔄 Optimizando {strategy.upper()}...")
                results = self.orchestrator.run_optimization(strategy, symbol)
                print(f"✅ {strategy.upper()} completado")
                
        except Exception as e:
            print(f"❌ Error en optimización: {e}")
    
    def view_optimization_results(self):
        """View optimization results"""
        print("📊 VER RESULTADOS DE OPTIMIZACIÓN")
        print("-" * 30)
        
        try:
            results = self.orchestrator.get_optimization_results()
            
            if not results:
                print("ℹ️ No hay resultados de optimización disponibles")
                return
            
            for strategy, data in results.items():
                print(f"\n📈 {strategy.upper()}:")
                print(f"  🎯 Score: {data.get('best_score', 'N/A')}")
                print(f"  ⚙️ Parámetros: {data.get('best_params', {})}")
                
        except Exception as e:
            print(f"❌ Error al obtener resultados: {e}")
    
    def hyperparameter_analysis(self):
        """Analyze hyperparameters"""
        print("🔬 ANÁLISIS DE HIPERPARÁMETROS")
        print("-" * 30)
        
        strategy = self.get_strategy_input()
        
        try:
            analysis = self.optimizer.analyze_hyperparameters(strategy)
            print(f"\n📊 Análisis para {strategy.upper()}:")
            print(analysis)
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
    
    # Signal Generation Methods
    def generate_all_signals(self):
        """Generate signals for all strategies"""
        print("📈 GENERAR SEÑALES (TODAS LAS ESTRATEGIAS)")
        print("-" * 30)
        
        symbol = self.get_symbol_input()
        
        try:
            for strategy in self.available_strategies:
                print(f"\n🔄 Generando señales {strategy.upper()}...")
                signals = self.signal_manager.generate_signals(strategy, symbol)
                print(f"✅ {len(signals)} señales generadas para {strategy.upper()}")
                
        except Exception as e:
            print(f"❌ Error al generar señales: {e}")
    
    def generate_specific_signals(self):
        """Generate signals for specific strategy"""
        print("📈 GENERAR SEÑALES (ESTRATEGIA ESPECÍFICA)")
        print("-" * 30)
        
        strategy = self.get_strategy_input()
        symbol = self.get_symbol_input()
        
        try:
            print(f"\n🔄 Generando señales {strategy.upper()} para {symbol}...")
            signals = self.signal_manager.generate_signals(strategy, symbol)
            print(f"✅ {len(signals)} señales generadas")
            
            # Show recent signals
            if len(signals) > 0:
                print("\n📊 Últimas 5 señales:")
                recent = signals.tail(5)
                for _, signal in recent.iterrows():
                    print(f"  {signal['timestamp']} - {signal['signal']} ({signal['confidence']:.2f})")
            
        except Exception as e:
            print(f"❌ Error al generar señales: {e}")
    
    def view_signals_status(self):
        """View signals status"""
        print("📊 ESTADO DE SEÑALES GENERADAS")
        print("-" * 30)
        
        try:
            status = self.signal_manager.get_signals_status()
            
            for strategy, data in status.items():
                print(f"\n📈 {strategy.upper()}:")
                print(f"  📅 Última actualización: {data.get('last_update', 'N/A')}")
                print(f"  🎯 Total señales: {data.get('total_signals', 0)}")
                print(f"  📊 Señales recientes: {data.get('recent_signals', 0)}")
                
        except Exception as e:
            print(f"❌ Error al obtener estado: {e}")
    
    def export_signals(self):
        """Export signals to CSV"""
        print("💾 EXPORTAR SEÑALES A CSV")
        print("-" * 30)
        
        strategy = self.get_strategy_input()
        symbol = self.get_symbol_input()
        
        output_path = input("\nRuta de salida (default: signals_export.csv): ").strip()
        if not output_path:
            output_path = "signals_export.csv"
        
        try:
            self.signal_manager.export_signals(strategy, symbol, output_path)
            print(f"✅ Señales exportadas a: {output_path}")
            
        except Exception as e:
            print(f"❌ Error al exportar señales: {e}")
    
    # Evaluation Methods
    def evaluate_performance(self):
        """Evaluate complete performance"""
        print("📊 EVALUAR RENDIMIENTO COMPLETO")
        print("-" * 30)
        
        try:
            results = self.orchestrator.run_full_evaluation()
            
            print("\n📈 Resultados de evaluación:")
            for strategy, metrics in results.items():
                print(f"\n🎯 {strategy.upper()}:")
                print(f"  💰 Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
                print(f"  📈 Total Return: {metrics.get('total_return', 'N/A'):.2%}")
                print(f"  📉 Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
                print(f"  🎯 Win Rate: {metrics.get('win_rate', 'N/A'):.2%}")
                
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
    
    def compare_strategies(self):
        """Compare strategies"""
        print("🔄 COMPARAR ESTRATEGIAS")
        print("-" * 30)
        
        try:
            comparison = self.orchestrator.compare_strategies()
            
            print("\n📊 Comparación de estrategias:")
            print(comparison)
            
        except Exception as e:
            print(f"❌ Error en comparación: {e}")
    
    def risk_analysis(self):
        """Risk analysis"""
        print("⚠️ ANÁLISIS DE RIESGO")
        print("-" * 30)
        
        try:
            risk_report = self.orchestrator.analyze_risk()
            print("\n📊 Reporte de riesgo:")
            print(risk_report)
            
        except Exception as e:
            print(f"❌ Error en análisis de riesgo: {e}")
    
    def metrics_report(self):
        """Generate metrics report"""
        print("📋 REPORTE DE MÉTRICAS")
        print("-" * 30)
        
        try:
            report = self.orchestrator.generate_metrics_report()
            print("\n📊 Reporte de métricas generado:")
            print(report)
            
        except Exception as e:
            print(f"❌ Error al generar reporte: {e}")
    
    # Utility Methods
    def view_config(self):
        """View current configuration"""
        print("⚙️ CONFIGURACIÓN ACTUAL")
        print("-" * 30)
        
        try:
            config = self.orchestrator.config_service.get_config()
            
            print("\n📝 Configuración del sistema:")
            for section, values in config.items():
                print(f"\n[{section}]")
                if isinstance(values, dict):
                    for key, value in values.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {values}")
                    
        except Exception as e:
            print(f"❌ Error al obtener configuración: {e}")
    
    def clean_temp_files(self):
        """Clean temporary files"""
        print("🧹 LIMPIAR ARCHIVOS TEMPORALES")
        print("-" * 30)
        
        try:
            cleaned = self.orchestrator.clean_temporary_files()
            print(f"✅ {cleaned} archivos temporales eliminados")
            
        except Exception as e:
            print(f"❌ Error al limpiar archivos: {e}")
    
    def run_diagnostics(self):
        """Run system diagnostics"""
        print("🔧 EJECUTAR DIAGNÓSTICOS")
        print("-" * 30)
        
        try:
            diagnostics = self.orchestrator.run_diagnostics()
            
            print("\n🔍 Resultados del diagnóstico:")
            for check, result in diagnostics.items():
                status_icon = "✅" if result['status'] else "❌"
                print(f"  {status_icon} {check}: {result['message']}")
                
        except Exception as e:
            print(f"❌ Error en diagnósticos: {e}")
    
    def system_status(self):
        """Show system status"""
        print("📊 ESTADO DEL SISTEMA")
        print("-" * 30)
        
        try:
            status = self.orchestrator.get_system_status()
            
            print("\n📈 Estado del sistema:")
            print(f"  🔧 Sistema: {status.get('system_health', 'Unknown')}")
            print(f"  📊 Datos: {status.get('data_status', 'Unknown')}")
            print(f"  🎯 Modelos: {status.get('models_status', 'Unknown')}")
            print(f"  📈 Señales: {status.get('signals_status', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error al obtener estado: {e}")
    
    def backup_config(self):
        """Backup configuration"""
        print("💾 BACKUP DE CONFIGURACIÓN")
        print("-" * 30)
        
        backup_name = input("\nNombre del backup (default: auto): ").strip()
        
        try:
            backup_path = self.orchestrator.backup_configuration(backup_name)
            print(f"✅ Configuración respaldada en: {backup_path}")
            
        except Exception as e:
            print(f"❌ Error al crear backup: {e}")
    
    def load_symbols_from_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('trading', {}).get('symbols', [])

    def load_strategies_from_dir(self):
        strategies_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies')
        strategies = []
        for entry in os.listdir(strategies_dir):
            entry_path = os.path.join(strategies_dir, entry)
            # Solo directorios y que tengan un strategy.py dentro
            if (
                os.path.isdir(entry_path)
                and not entry.startswith('__')
                and os.path.exists(os.path.join(entry_path, 'strategy.py'))
            ):
                strategies.append(entry)
        return strategies

    def show_cache_status_for_symbol(self):
        """Mostrar estado de la caché para un símbolo (todas las temporalidades)"""
        print("\n=== Estado de la caché ===")
        symbol = self.get_symbol_input("Seleccione símbolo para comprobar la caché")
        # Puedes obtener los timeframes del config o definirlos aquí
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        print(f"\nEstado de la caché para {symbol}:")
        for tf in timeframes:
            try:
                data = self.data_pipeline.fetch_data(
                    symbol=symbol, timeframe=tf, allow_download=False
                )
                if data is None or data.data.empty:
                    print(f"❌ Sin datos en caché para {tf}")
                else:
                    print(f"✅ Datos en caché para {tf} ({len(data.data)} registros)")
            except Exception as e:
                print(f"❌ Error al consultar {tf}: {e}")

    def visualizar_datos_historicos(self):
        """Visualizar datos históricos con gráficos y alineación temporal"""
        print("📈 VISUALIZAR DATOS HISTÓRICOS")
        print("-" * 30)
        symbol = self.get_symbol_input()
        timeframes = ['M15', 'H1', 'H4']
        data = {}
        for tf in timeframes:
            try:
                result = self.data_pipeline.fetch_data(symbol, tf, allow_download=False)
                if result and not result.data.empty:
                    data[tf] = result.data
            except Exception as e:
                print(f"❌ Error al cargar {tf}: {e}")
        if not data:
            print("❌ No hay datos para graficar.")
            return
        # Mostrar alineación temporal
        from src.utils.utils import chequear_alineacion_temporal
        chequear_alineacion_temporal(data)
        # Preguntar si quiere ver EMAs
        mostrar_ema = input("¿Mostrar EMAs en el gráfico? (s/n): ").strip().lower() == 's'
        ema_fast = 12
        ema_slow = 26
        if mostrar_ema:
            try:
                ema_fast = int(input("EMA rápida (default 12): ") or 12)
                ema_slow = int(input("EMA lenta (default 26): ") or 26)
            except ValueError:
                print("Valores no válidos, usando 12 y 26.")
        try:
            from src.utils.utils import graficar_datos
            graficar_datos(data, mostrar_ema=mostrar_ema, ema_fast=ema_fast, ema_slow=ema_slow)
        except Exception as e:
            print(f"❌ Error al graficar datos: {e}")

def main():
    """Main entry point for the menu interface"""
    try:
        menu = MenuInterface()
        menu.display_main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

def importar_mt5_historico(pipeline, symbol: str):
    """
    Importa datos históricos de MT5 para un símbolo, usando rangos según la temporalidad.
    - M1, M5, M15, M30: mes a mes desde 2019 hasta hoy.
    - H1, H4, D1: año a año desde 2015 hasta hoy.
    """
    timeframes_mes = ['M1', 'M5', 'M15', 'M30']
    timeframes_ano = ['H1', 'H4', 'D1']
    hoy = datetime.now()

    # Rango mes a mes para timeframes de minutos
    for tf in timeframes_mes:
        start = datetime(2019, 1, 1)
        while start < hoy:
            end = (start + timedelta(days=32)).replace(day=1)
            if end > hoy:
                end = hoy
            print(f"Importando {symbol} {tf} desde {start.date()} hasta {end.date()}")
            try:
                pipeline.fetch_data(
                    symbol=symbol,
                    timeframe=tf,
                    provider='mt5',
                    start_date=start,
                    end_date=end,
                    allow_download=True
                )
            except Exception as e:
                print(f"Error importando {symbol} {tf} {start.date()} - {end.date()}: {e}")
            start = end

    # Rango año a año para timeframes mayores
    for tf in timeframes_ano:
        start = datetime(2015, 1, 1)
        while start < hoy:
            end = datetime(start.year + 1, 1, 1)
            if end > hoy:
                end = hoy
            print(f"Importando {symbol} {tf} desde {start.date()} hasta {end.date()}")
            try:
                pipeline.fetch_data(
                    symbol=symbol,
                    timeframe=tf,
                    provider='mt5',
                    start_date=start,
                    end_date=end,
                    allow_download=True
                )
            except Exception as e:
                print(f"Error importando {symbol} {tf} {start.date()} - {end.date()}: {e}")
            start = end

    pipeline.check_cache_integrity('EURUSD', 'M15', 2024)
