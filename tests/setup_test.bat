#!/bin/bash
echo "🔧 Installing test dependencies..."

# Install from test requirements
pip install -r tests/test_requirements.txt

# Install additional performance testing tools
pip install pytest-benchmark
pip install pytest-xdist
pip install psutil

echo "✅ Test dependencies installed successfully!"
echo ""
echo "Available test tools:"
echo "- pytest: Main testing framework"
echo "- pytest-html: HTML reports"
echo "- pytest-cov: Coverage analysis"  
echo "- pytest-benchmark: Performance benchmarking"
echo "- pytest-xdist: Parallel testing"
echo "- psutil: Memory/CPU monitoring"
echo ""
echo "Run tests with: python tests/test_menu.py"