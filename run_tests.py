"""Script to run all tests with coverage"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests with coverage reporting"""
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Install test dependencies
    print("🔧 Installing test dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'tests/requirements.txt'])
    
    print("\n🧪 Running tests with coverage...")
    
    test_commands = [
        # Unit tests first (most likely to work)
        [sys.executable, '-m', 'pytest', 'tests/unit/', '-v', '--cov=src', '--cov-report=html', '--cov-report=term'],
    ]
    
    # ✅ Corregir ruta: usar 'services' (plural)
    if os.path.exists('src/services/orchestrator.py'):
        print("✅ Found real orchestrator in src/services/, including integration tests")
        test_commands.extend([
            # Integration tests
            [sys.executable, '-m', 'pytest', 'tests/integration/', '-v'],
            
            # Performance tests
            [sys.executable, '-m', 'pytest', 'tests/performance/', '-v', '--benchmark-only']
        ])
    else:
        print("ℹ️ Orchestrator not found in src/services/, skipping integration tests")
        print("📍 Expected location: src/services/orchestrator.py")
        print("📂 Current structure check:")
        
        # Show what actually exists
        if os.path.exists('src/'):
            print("   📁 src/ exists")
            subdirs = [d for d in os.listdir('src/') if os.path.isdir(os.path.join('src/', d))]
            for subdir in subdirs:
                print(f"   📁 src/{subdir}/")
                if subdir == 'services' and os.path.exists(f'src/{subdir}/'):
                    files = [f for f in os.listdir(f'src/{subdir}/') if f.endswith('.py')]
                    for file in files:
                        print(f"      📄 {file}")
    
    success = True
    for cmd in test_commands:
        print(f"\n📋 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"❌ Test failed with code {result.returncode}")
            success = False
    
    if success:
        print("\n✅ All tests completed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n⚠️ Some tests failed")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)