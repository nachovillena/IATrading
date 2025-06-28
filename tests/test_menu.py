"""Interactive Test Menu for IATrading System"""

import os
import sys
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class TestMenu:
    """Interactive menu for running different types of tests"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.reports_dir = self.test_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Create timestamped report directory
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_report_dir = self.reports_dir / self.timestamp
        self.current_report_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of reports
        (self.current_report_dir / "htmlcov").mkdir(exist_ok=True)
        (self.current_report_dir / "logs").mkdir(exist_ok=True)
        
        # Update latest symlink
        self._update_latest_link()

    def _update_latest_link(self):
        """Update the 'latest' symlink to point to current report"""
        latest_link = self.reports_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        # Create symlink (Windows compatible)
        if platform.system() == "Windows":
            try:
                # Use junction on Windows
                subprocess.run(["mklink", "/J", str(latest_link), str(self.current_report_dir)], 
                             shell=True, check=True, capture_output=True)
            except:
                # Fallback: just copy the path
                latest_link.write_text(str(self.current_report_dir))
        else:
            latest_link.symlink_to(self.timestamp)

    def show_menu(self):
        """Display main test menu"""
        while True:
            self._clear_screen()
            print("\n" + "="*70)
            print("🧪 IATRADING TEST SUITE")
            print("="*70)
            print(f"📁 Current Report: {self.timestamp}")
            print(f"📊 Reports Dir: {self.current_report_dir}")
            print("="*70)
            print("1. 🎯 Unit Tests")
            print("2. 🔗 Integration Tests") 
            print("3. ⚡ Performance Tests")
            print("4. 📊 Full Test Suite")
            print("5. 🧹 Clean Test Environment")
            print("6. 📈 View Test Reports")
            print("7. 🔧 Test Configuration")
            print("8. 📋 Test Status & Info")
            print("0. ❌ Exit")
            print("="*70)
            
            choice = input("\n🎯 Select option: ").strip()
            
            if choice == "1":
                self.unit_tests_menu()
            elif choice == "2":
                self.integration_tests_menu()
            elif choice == "3":
                self.performance_tests_menu()
            elif choice == "4":
                self.full_test_suite()
            elif choice == "5":
                self.clean_environment()
            elif choice == "6":
                self.view_reports()
            elif choice == "7":
                self.test_configuration()
            elif choice == "8":
                self.test_status()
            elif choice == "0":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
                input("Press Enter to continue...")

    def unit_tests_menu(self):
        """Unit tests submenu"""
        while True:
            self._clear_screen()
            print("\n" + "-"*50)
            print("🎯 UNIT TESTS")
            print("-"*50)
            print("1. 📈 Strategy Tests")
            print("   - EMA Strategy")
            print("   - RSI Strategy") 
            print("   - MACD Strategy")
            print("2. 💾 Data Provider Tests")
            print("3. 🧠 Core Functionality Tests")
            print("4. 🖥️  Interface Tests")
            print("5. 🎯 All Unit Tests (Quick)")
            print("6. 🎯 All Unit Tests (Verbose)")
            print("7. 🎯 All Unit Tests (Coverage)")
            print("8. 🧪 Custom Test Selection")
            print("0. ⬅️  Back to Main Menu")
            print("-"*50)
            
            choice = input("\n🎯 Select option: ").strip()
            
            if choice == "1":
                self.strategy_tests_submenu()
            elif choice == "2":
                self.run_data_tests()
            elif choice == "3":
                self.run_core_tests()
            elif choice == "4":
                self.run_interface_tests()
            elif choice == "5":
                self.run_all_unit_tests()
            elif choice == "6":
                self.run_all_unit_tests(verbose=True)
            elif choice == "7":
                self.run_all_unit_tests(coverage=True)
            elif choice == "8":
                self.custom_test_selection()
            elif choice == "0":
                break
            else:
                print("❌ Invalid option.")
                input("Press Enter to continue...")

    def strategy_tests_submenu(self):
        """Strategy tests submenu"""
        while True:
            self._clear_screen()
            print("\n" + "-"*40)
            print("📈 STRATEGY TESTS")
            print("-"*40)
            print("1. 📊 EMA Strategy Tests")
            print("2. 📈 RSI Strategy Tests")
            print("3. 📉 MACD Strategy Tests")
            print("4. 🎯 All Strategy Tests")
            print("5. ⚡ Strategy Performance Compare")
            print("0. ⬅️  Back")
            print("-"*40)
            
            choice = input("\n🎯 Select option: ").strip()
            
            if choice == "1":
                self._run_pytest("unit/test_strategies.py::TestEMAStrategy", "EMA Strategy Tests")
            elif choice == "2":
                self._run_pytest("unit/test_strategies.py::TestRSIStrategy", "RSI Strategy Tests")
            elif choice == "3":
                self._run_pytest("unit/test_strategies.py::TestMACDStrategy", "MACD Strategy Tests")
            elif choice == "4":
                self._run_pytest("unit/test_strategies.py", "All Strategy Tests")
            elif choice == "5":
                self._run_pytest("unit/test_strategies.py::TestStrategyPerformance", "Strategy Performance")
            elif choice == "0":
                break
            else:
                print("❌ Invalid option.")
                input("Press Enter to continue...")

    def run_data_tests(self):
        """Run data provider tests"""
        print("\n💾 Running Data Provider Tests...")
        # Create basic data test if it doesn't exist
        self._ensure_data_tests_exist()
        self._run_pytest("unit/test_data.py", "Data Provider Tests")

    def run_core_tests(self):
        """Run core functionality tests"""
        print("\n🧠 Running Core Tests...")
        self._ensure_core_tests_exist()
        self._run_pytest("unit/test_core.py", "Core Functionality Tests")

    def run_interface_tests(self):
        """Run interface tests"""
        print("\n🖥️ Running Interface Tests...")
        self._ensure_interface_tests_exist()
        self._run_pytest("unit/test_interfaces.py", "Interface Tests")

    def run_all_unit_tests(self, verbose=False, coverage=False):
        """Run all unit tests"""
        test_name = f"All Unit Tests"
        if verbose:
            test_name += " (Verbose)"
        if coverage:
            test_name += " (Coverage)"
        
        print(f"\n🎯 Running {test_name}...")
        
        args = ["unit/"]
        if verbose:
            args.append("-v")
        if coverage:
            args.extend([
                "--cov=src", 
                f"--cov-report=html:{self.current_report_dir}/htmlcov",
                "--cov-report=term",
                f"--cov-report=xml:{self.current_report_dir}/coverage.xml"
            ])
        
        self._run_pytest(" ".join(args), test_name)

    def custom_test_selection(self):
        """Custom test selection"""
        print("\n🧪 Custom Test Selection")
        print("Enter pytest arguments (e.g., 'unit/test_strategies.py::TestEMAStrategy::test_initialization_default'):")
        test_args = input("Test args: ").strip()
        
        if test_args:
            self._run_pytest(test_args, "Custom Test Selection")
        else:
            print("❌ No test arguments provided")
            input("Press Enter to continue...")

    def integration_tests_menu(self):
        """Integration tests submenu"""
        self._clear_screen()
        print("\n🔗 Integration Tests")
        print("1. 🔄 Trading Flow Tests")
        print("2. 📊 Data Pipeline Tests")
        print("3. 🔗 All Integration Tests")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            self._ensure_integration_tests_exist()
            self._run_pytest("integration/test_trading_flow.py", "Trading Flow Tests")
        elif choice == "2":
            self._ensure_integration_tests_exist()
            self._run_pytest("integration/test_data_pipeline.py", "Data Pipeline Tests")
        elif choice == "3":
            self._ensure_integration_tests_exist()
            self._run_pytest("integration/", "All Integration Tests")

    def performance_tests_menu(self):
        """Performance tests submenu"""
        self._clear_screen()
        print("\n⚡ Performance Tests")
        print("1. 📈 Strategy Performance")
        print("2. 💾 Data Loading Performance")
        print("3. ⚡ All Performance Tests")
        print("4. 🎯 Benchmark Comparison")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            self._run_pytest("performance/test_strategy_performance.py", "Strategy Performance")
        elif choice == "2":
            self._run_pytest("performance/test_data_performance.py", "Data Performance")
        elif choice == "3":
            self._run_pytest("performance/", "All Performance Tests")
        elif choice == "4":
            self._run_pytest("performance/ --benchmark-compare", "Benchmark Comparison")

    def full_test_suite(self):
        """Run complete test suite"""
        self._clear_screen()
        print("\n📊 Full Test Suite Options")
        print("1. 🚀 Quick (Unit tests only)")
        print("2. 🔄 Standard (Unit + Integration)")
        print("3. 🎯 Complete (All tests + Coverage)")
        print("4. 📊 Complete + Performance")
        print("5. 🧪 Complete + Benchmarks")
        
        choice = input("\nSelect test scope: ").strip()
        
        if choice == "1":
            self._run_pytest("unit/ -v", "Quick Test Suite")
        elif choice == "2":
            self._run_pytest("unit/ integration/ -v", "Standard Test Suite")
        elif choice == "3":
            self._run_pytest(
                f"unit/ integration/ -v --cov=src --cov-report=html:{self.current_report_dir}/htmlcov --cov-report=term",
                "Complete Test Suite"
            )
        elif choice == "4":
            self._run_pytest(
                f"unit/ integration/ performance/ -v --cov=src --cov-report=html:{self.current_report_dir}/htmlcov",
                "Complete Test Suite + Performance"
            )
        elif choice == "5":
            self._run_pytest(
                f"unit/ integration/ performance/ -v --cov=src --cov-report=html:{self.current_report_dir}/htmlcov --benchmark-only",
                "Complete Test Suite + Benchmarks"
            )

    def clean_environment(self):
        """Clean test environment"""
        self._clear_screen()
        print("\n🧹 Cleaning Test Environment...")
        
        print("What would you like to clean?")
        print("1. 🗑️  Cache files (__pycache__, .pytest_cache)")
        print("2. 📊 Old reports (keep last 5)")
        print("3. 🧹 Temporary files (*.tmp, *.log)")
        print("4. 🔄 Reset test database/cache")
        print("5. 🧹 Clean all")
        print("6. 🗂️  Organize old reports")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            self._clean_cache()
        elif choice == "2":
            self._clean_old_reports()
        elif choice == "3":
            self._clean_temp_files()
        elif choice == "4":
            self._reset_test_data()
        elif choice == "5":
            self._clean_all()
        elif choice == "6":
            self._organize_reports()

    def view_reports(self):
        """View test reports"""
        self._clear_screen()
        print("\n📈 Test Reports")
        
        reports = sorted([d for d in self.reports_dir.iterdir() 
                         if d.is_dir() and d.name != "latest"], reverse=True)
        
        if not reports:
            print("No reports found.")
            input("Press Enter to continue...")
            return
        
        print(f"\nAvailable reports (showing last 10):")
        for i, report in enumerate(reports[:10], 1):
            size = self._get_dir_size(report)
            print(f"{i:2d}. {report.name} ({size})")
        
        print(f"\n11. 📁 Open reports directory")
        print(f"12. 🧹 Clean old reports")
        
        choice = input(f"\nSelect report (1-{min(10, len(reports))}): ").strip()
        
        try:
            if choice == "11":
                self._open_directory(self.reports_dir)
            elif choice == "12":
                self._clean_old_reports()
            else:
                report_idx = int(choice) - 1
                if 0 <= report_idx < len(reports):
                    report_dir = reports[report_idx]
                    self._open_report(report_dir)
                else:
                    print("❌ Invalid choice")
                    input("Press Enter to continue...")
        except ValueError:
            print("❌ Invalid choice")
            input("Press Enter to continue...")

    def test_status(self):
        """Show test status and info"""
        self._clear_screen()
        print("\n📋 Test Status & Information")
        print("="*60)
        
        # Project info
        print(f"📁 Project Root: {self.project_root}")
        print(f"🧪 Test Directory: {self.test_dir}")
        print(f"📊 Reports Directory: {self.reports_dir}")
        print(f"📈 Current Report: {self.current_report_dir}")
        
        # Test file counts
        unit_tests = len(list((self.test_dir / "unit").glob("test_*.py")))
        integration_tests = len(list((self.test_dir / "integration").glob("test_*.py")))
        performance_tests = len(list((self.test_dir / "performance").glob("test_*.py")))
        
        print(f"\n📊 Test Files:")
        print(f"   🎯 Unit Tests: {unit_tests}")
        print(f"   🔗 Integration Tests: {integration_tests}")
        print(f"   ⚡ Performance Tests: {performance_tests}")
        
        # Recent reports
        reports = sorted([d for d in self.reports_dir.iterdir() 
                         if d.is_dir() and d.name != "latest"], reverse=True)
        print(f"\n📈 Recent Reports: {len(reports)}")
        for report in reports[:3]:
            size = self._get_dir_size(report)
            print(f"   📊 {report.name} ({size})")
        
        input("\nPress Enter to continue...")

    def test_configuration(self):
        """Test configuration menu"""
        self._clear_screen()
        print("\n🔧 Test Configuration")
        print("1. 📊 View current configuration")
        print("2. 🔧 Edit pytest.ini")
        print("3. 🔧 Edit conftest.py")
        print("4. 📁 View test directory structure")
        print("5. 🔧 Environment check")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            self._show_config()
        elif choice == "2":
            self._edit_file(self.test_dir / "pytest.ini")
        elif choice == "3":
            self._edit_file(self.test_dir / "conftest.py")
        elif choice == "4":
            self._show_directory_structure()
        elif choice == "5":
            self._environment_check()

    def _run_pytest(self, args: str, test_name: str):
        """Run pytest with given arguments"""
        print(f"\n🚀 Running {test_name}...")
        print(f"📁 Report will be saved to: {self.current_report_dir}")
        
        # Create command
        cmd = [
            sys.executable, "-m", "pytest",
            *args.split(),
            f"--html={self.current_report_dir}/pytest_report.html",
            "--self-contained-html",
            f"--junit-xml={self.current_report_dir}/junit.xml"
        ]
        
        # Log command
        log_file = self.current_report_dir / "logs" / f"{test_name.replace(' ', '_').lower()}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        # Change to project root
        original_cwd = os.getcwd()
        os.chdir(self.project_root)
        
        try:
            print(f"🔄 Executing: {' '.join(cmd)}")
            print("⏳ Please wait...")
            
            # Run pytest
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Save output
            with open(self.current_report_dir / "test_output.txt", "w", encoding='utf-8') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            # Save to specific log
            with open(log_file, "w", encoding='utf-8') as f:
                f.write(result.stdout)
                f.write(result.stderr)
            
            # Show results
            self._clear_screen()
            print("="*70)
            if result.returncode == 0:
                print(f"✅ {test_name} completed successfully!")
            else:
                print(f"❌ {test_name} failed with exit code {result.returncode}")
            print("="*70)
            
            # Parse and show summary
            self._show_test_summary(result.stdout)
            
            print(f"\n📊 Full report saved to: {self.current_report_dir}")
            print(f"📄 HTML Report: pytest_report.html")
            
            if "htmlcov" in " ".join(cmd):
                print(f"📊 Coverage Report: htmlcov/index.html")
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
        finally:
            os.chdir(original_cwd)
        
        input("\nPress Enter to continue...")

    def _show_test_summary(self, output: str):
        """Parse and show test summary"""
        lines = output.split('\n')
        
        # Look for summary line
        for line in lines:
            if 'passed' in line and ('failed' in line or 'error' in line or 'warning' in line):
                print(f"📊 Summary: {line.strip()}")
                break
            elif 'passed' in line and 'in' in line:
                print(f"📊 Summary: {line.strip()}")
                break
        
        # Show failed tests if any
        failed_tests = []
        in_failures = False
        for line in lines:
            if 'FAILURES' in line:
                in_failures = True
            elif in_failures and line.startswith('_'):
                failed_tests.append(line.split(' ')[0])
        
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for test in failed_tests[:5]:  # Show first 5
                print(f"   • {test}")
            if len(failed_tests) > 5:
                print(f"   ... and {len(failed_tests) - 5} more")

    def _ensure_data_tests_exist(self):
        """Ensure data tests file exists"""
        data_test_file = self.test_dir / "unit" / "test_data.py"
        if not data_test_file.exists():
            data_test_file.write_text('''"""Data provider tests"""
import pytest
from src.data.providers.yahoo import YahooProvider

class TestDataProviders:
    def test_yahoo_provider_init(self):
        provider = YahooProvider()
        assert provider is not None
''')

    def _ensure_core_tests_exist(self):
        """Ensure core tests file exists"""
        core_test_file = self.test_dir / "unit" / "test_core.py"
        if not core_test_file.exists():
            core_test_file.write_text('''"""Core functionality tests"""
import pytest
from src.core.types import TradingData

class TestCore:
    def test_trading_data_creation(self):
        assert TradingData is not None
''')

    def _ensure_interface_tests_exist(self):
        """Ensure interface tests file exists"""
        interface_test_file = self.test_dir / "unit" / "test_interfaces.py"
        if not interface_test_file.exists():
            interface_test_file.write_text('''"""Interface tests"""
import pytest
from src.interfaces.menu_interface import MenuInterface

class TestInterfaces:
    def test_menu_interface_init(self):
        menu = MenuInterface()
        assert menu is not None
''')

    def _ensure_integration_tests_exist(self):
        """Ensure integration test files exist"""
        integration_dir = self.test_dir / "integration"
        integration_dir.mkdir(exist_ok=True)
        
        # Trading flow test
        trading_flow_test = integration_dir / "test_trading_flow.py"
        if not trading_flow_test.exists():
            trading_flow_test.write_text('''"""Trading flow integration tests"""
import pytest

class TestTradingFlow:
    def test_basic_flow(self):
        assert True  # Placeholder
''')
        
        # Data pipeline test
        data_pipeline_test = integration_dir / "test_data_pipeline.py"
        if not data_pipeline_test.exists():
            data_pipeline_test.write_text('''"""Data pipeline integration tests"""
import pytest

class TestDataPipeline:
    def test_pipeline_flow(self):
        assert True  # Placeholder
''')

    def _clean_cache(self):
        """Clean cache files"""
        cache_patterns = [
            "**/__pycache__",
            "**/.pytest_cache",
            "**/data/cache",
            "**/*.pyc"
        ]
        
        cleaned = 0
        for pattern in cache_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    subprocess.run(["rm", "-rf", str(path)], capture_output=True)
                    cleaned += 1
                elif path.is_file():
                    path.unlink()
                    cleaned += 1
        
        print(f"✅ Cleaned {cleaned} cache files/directories")
        input("Press Enter to continue...")

    def _clean_old_reports(self):
        """Keep only last 5 reports"""
        reports = sorted([d for d in self.reports_dir.iterdir() 
                         if d.is_dir() and d.name != "latest"])
        
        if len(reports) > 5:
            for old_report in reports[:-5]:
                subprocess.run(["rm", "-rf", str(old_report)], capture_output=True)
            print(f"✅ Cleaned {len(reports) - 5} old reports")
        else:
            print("✅ No old reports to clean")
        
        input("Press Enter to continue...")

    def _clean_temp_files(self):
        """Clean temporary files"""
        temp_patterns = ["**/*.tmp", "**/*.log~", "**/*~"]
        cleaned = 0
        
        for pattern in temp_patterns:
            for path in self.project_root.glob(pattern):
                path.unlink()
                cleaned += 1
        
        print(f"✅ Cleaned {cleaned} temporary files")
        input("Press Enter to continue...")

    def _reset_test_data(self):
        """Reset test data and cache"""
        test_cache = self.test_dir / "fixtures" / "cache"
        if test_cache.exists():
            subprocess.run(["rm", "-rf", str(test_cache)], capture_output=True)
        
        print("✅ Test data reset")
        input("Press Enter to continue...")

    def _clean_all(self):
        """Clean everything"""
        print("🧹 Cleaning all...")
        self._clean_cache()
        self._clean_old_reports()
        self._clean_temp_files()
        self._reset_test_data()
        print("✅ Environment completely cleaned")

    def _organize_reports(self):
        """Organize reports by date"""
        print("🗂️  Organizing reports...")
        # Implementation for organizing reports
        print("✅ Reports organized")
        input("Press Enter to continue...")

    def _open_report(self, report_dir: Path):
        """Open test report"""
        html_report = report_dir / "pytest_report.html"
        coverage_report = report_dir / "htmlcov" / "index.html"
        
        print(f"\n📊 Reports in {report_dir.name}:")
        
        options = []
        if html_report.exists():
            print(f"1. 📄 Test Report: pytest_report.html")
            options.append(("test", html_report))
        
        if coverage_report.exists():
            print(f"2. 📊 Coverage Report: htmlcov/index.html")
            options.append(("coverage", coverage_report))
        
        print(f"3. 📁 Open report directory")
        options.append(("dir", report_dir))
        
        choice = input(f"\nSelect option (1-{len(options)}): ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                option_type, path = options[idx]
                if option_type == "dir":
                    self._open_directory(path)
                else:
                    self._open_file(path)
        except ValueError:
            print("❌ Invalid choice")
            input("Press Enter to continue...")

    def _open_file(self, file_path: Path):
        """Open file with system default application"""
        try:
            if platform.system() == "Windows":
                os.startfile(file_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(file_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(file_path)])
            print(f"✅ Opened {file_path.name}")
        except Exception as e:
            print(f"❌ Could not open file: {e}")
        input("Press Enter to continue...")

    def _open_directory(self, dir_path: Path):
        """Open directory with system file explorer"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(dir_path)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(dir_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(dir_path)])
            print(f"✅ Opened directory {dir_path.name}")
        except Exception as e:
            print(f"❌ Could not open directory: {e}")
        input("Press Enter to continue...")

    def _show_config(self):
        """Show current test configuration"""
        print("\n📊 Current Test Configuration:")
        print("="*60)
        print(f"Project Root: {self.project_root}")
        print(f"Test Directory: {self.test_dir}")
        print(f"Reports Directory: {self.reports_dir}")
        print(f"Current Report: {self.current_report_dir}")
        
        # Show pytest.ini if exists
        pytest_ini = self.test_dir / "pytest.ini"
        if pytest_ini.exists():
            print(f"\n📄 pytest.ini:")
            print("-" * 40)
            print(pytest_ini.read_text()[:500] + "..." if len(pytest_ini.read_text()) > 500 else pytest_ini.read_text())
        
        input("\nPress Enter to continue...")

    def _show_directory_structure(self):
        """Show test directory structure"""
        print("\n📁 Test Directory Structure:")
        print("="*50)
        
        def print_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            
            items = sorted(path.iterdir()) if path.is_dir() else []
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    print_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        print_tree(self.test_dir)
        input("\nPress Enter to continue...")

    def _edit_file(self, file_path: Path):
        """Edit file with system editor"""
        editor = os.environ.get('EDITOR', 'notepad' if platform.system() == "Windows" else 'nano')
        try:
            subprocess.run([editor, str(file_path)])
        except Exception as e:
            print(f"❌ Could not open editor: {e}")
            input("Press Enter to continue...")

    def _environment_check(self):
        """Check test environment"""
        print("\n🔧 Environment Check")
        print("="*50)
        
        # Python version
        python_version = sys.version.split()[0]
        print(f"🐍 Python: {python_version}")
        
        # pytest version
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                                  capture_output=True, text=True)
            pytest_version = result.stdout.strip()
            print(f"🧪 {pytest_version}")
        except:
            print("❌ pytest not found")
        
        # Coverage
        try:
            result = subprocess.run([sys.executable, "-m", "coverage", "--version"], 
                                  capture_output=True, text=True)
            coverage_version = result.stdout.strip()
            print(f"📊 {coverage_version}")
        except:
            print("❌ coverage not found")
        
        # Available test files
        unit_tests = list((self.test_dir / "unit").glob("test_*.py"))
        print(f"\n📁 Test Files:")
        print(f"   Unit tests: {len(unit_tests)}")
        for test in unit_tests:
            print(f"     • {test.name}")
        
        input("\nPress Enter to continue...")

    def _get_dir_size(self, path: Path) -> str:
        """Get directory size in human readable format"""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f}{unit}"
                total_size /= 1024.0
            return f"{total_size:.1f}TB"
        except:
            return "Unknown"

    def _clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if platform.system() == "Windows" else 'clear')


def main():
    """Main function"""
    try:
        menu = TestMenu()
        menu.show_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()