# 🔄 Integración CI/CD - Testing Pipeline

## 🎯 Estrategia de CI/CD para Testing

### Pipeline Overview
```
📋 Stages del Pipeline:
├── 🔍 Lint & Format Check
├── 🧪 Unit Tests (Rápidos)
├── 🔗 Integration Tests
├── 📊 Coverage Analysis  
├── 🚀 Performance Tests
└── 📈 Report Generation
```

## 🛠️ GitHub Actions Configuration

### Basic Workflow
```yaml
# filepath: .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Unit Tests
      run: |
        python -m pytest tests/unit/ -v --tb=short
    
    - name: Integration Tests
      run: |
        python -m pytest tests/integration/ -v --tb=short
    
    - name: Coverage Report
      run: |
        python -m pytest tests/ --cov=src --cov-report=xml --cov-fail-under=35
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Advanced Workflow con Stages
```yaml
# filepath: .github/workflows/advanced-test.yml
name: Advanced Test Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  # Stage 1: Linting y formato
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    - name: Install lint dependencies
      run: |
        pip install flake8 black isort
    - name: Run black
      run: black --check src/ tests/
    - name: Run isort
      run: isort --check-only src/ tests/
    - name: Run flake8
      run: flake8 src/ tests/

  # Stage 2: Tests rápidos (Unit)
  unit-tests:
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    - name: Run unit tests
      run: |
        python -m pytest tests/unit/ -v --tb=short -n auto
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: unit-test-results-${{ matrix.python-version }}
        path: tests/reports/

  # Stage 3: Integration Tests
  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/ -v --tb=short
    - name: Upload integration results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: tests/reports/

  # Stage 4: Coverage Analysis
  coverage:
    needs: [unit-tests, integration-tests]
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run coverage analysis
      run: |
        python -m pytest tests/ --cov=src --cov-report=xml --cov-report=html --cov-fail-under=35
    - name: Upload coverage reports
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: htmlcov/
    - name: Comment coverage on PR
      if: github.event_name == 'pull_request'
      uses: py-cov-action/python-coverage-comment-action@v3
      with:
        GITHUB_TOKEN: ${{ github.token }}

  # Stage 5: Performance Tests
  performance:
    needs: coverage
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-benchmark
    - name: Run performance tests
      run: |
        python -m pytest -k "performance" --benchmark-json=benchmark.json
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

## 🔧 GitLab CI Configuration

```yaml
# filepath: .gitlab-ci.yml
stages:
  - lint
  - test
  - coverage
  - performance
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

# Stage 1: Linting
lint:
  stage: lint
  image: python:3.9
  script:
    - pip install flake8 black isort
    - black --check src/ tests/
    - isort --check-only src/ tests/
    - flake8 src/ tests/
  only:
    - merge_requests
    - main

# Stage 2: Unit Tests
unit_tests:
  stage: test
  image: python:3.9
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
    - pip install pytest pytest-cov pytest-xdist
    - python -m pytest tests/unit/ -v --junitxml=report.xml
  artifacts:
    when: always
    reports:
      junit: report.xml
    paths:
      - tests/reports/
    expire_in: 1 week
  only:
    - merge_requests
    - main

# Stage 3: Integration Tests
integration_tests:
  stage: test
  image: python:3.9
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - python -m pytest tests/integration/ -v --junitxml=integration_report.xml
  artifacts:
    when: always
    reports:
      junit: integration_report.xml
    expire_in: 1 week
  only:
    - merge_requests
    - main

# Stage 4: Coverage
coverage:
  stage: coverage
  image: python:3.9
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - python -m pytest tests/ --cov=src --cov-report=xml --cov-report=html --cov-fail-under=35
    - echo "Coverage completed"
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week
  coverage: '/TOTAL.*\s+(\d+%)$/'
  only:
    - merge_requests
    - main

# Stage 5: Performance Tests
performance:
  stage: performance
  image: python:3.9
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
    - pip install pytest pytest-benchmark
    - python -m pytest -k "performance" --benchmark-json=benchmark.json
  artifacts:
    paths:
      - benchmark.json
    expire_in: 1 week
  only:
    - main
```

## 📊 Quality Gates

### Coverage Gates
```yaml
# En workflow de GitHub Actions
- name: Coverage Gate
  run: |
    coverage_percentage=$(python -m pytest --cov=src --cov-report=term-missing | grep "TOTAL" | awk '{print $4}' | sed 's/%//')
    echo "Current coverage: ${coverage_percentage}%"
    
    if [ $coverage_percentage -lt 35 ]; then
      echo "❌ Coverage ${coverage_percentage}% is below minimum 35%"
      exit 1
    else
      echo "✅ Coverage ${coverage_percentage}% meets requirement"
    fi
```

### Test Quality Gates
```yaml
- name: Test Quality Gate
  run: |
    # Ejecutar tests y capturar resultados
    python -m pytest tests/ --junitxml=results.xml --tb=short
    
    # Verificar que no hay tests skipped críticos
    skipped_count=$(grep -c "skipped" results.xml || true)
    if [ $skipped_count -gt 2 ]; then
      echo "❌ Too many skipped tests: $skipped_count"
      exit 1
    fi
    
    # Verificar tiempo de ejecución
    execution_time=$(grep "time=" results.xml | head -1 | sed 's/.*time="\([^"]*\)".*/\1/')
    echo "Execution time: ${execution_time}s"
```

## 🎯 Branch Protection Rules

### Main Branch Protection
```yaml
# GitHub Branch Protection Settings
- Require pull request reviews before merging
- Require status checks to pass before merging:
  - lint
  - unit-tests (python 3.8)
  - unit-tests (python 3.9)  
  - unit-tests (python 3.10)
  - integration-tests
  - coverage (>= 35%)
- Require branches to be up to date before merging
- Require linear history
```

### Development Workflow
```bash
# 1. Feature branch workflow
git checkout -b feature/new-strategy
git commit -m "Add new strategy"
git push origin feature/new-strategy

# 2. CI ejecuta automáticamente:
#    - Lint check
#    - Unit tests
#    - Integration tests
#    - Coverage analysis

# 3. PR merge solo si todos los checks pasan
```

## 📈 Monitoring y Alertas

### Slack Integration
```yaml
# En GitHub Actions
- name: Notify Slack on Failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#development'
    text: |
      🚨 Test Suite Failed
      Branch: ${{ github.ref }}
      Commit: ${{ github.sha }}
      Author: ${{ github.actor }}
```

### Email Notifications
```yaml
# GitLab CI notification
after_script:
  - |
    if [ "$CI_JOB_STATUS" == "failed" ]; then
      echo "Sending failure notification..."
      # Script para enviar email
    fi
```

## 🔄 Automated Tasks

### Nightly Tests
```yaml
# .github/workflows/nightly.yml
name: Nightly Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily

jobs:
  comprehensive-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-benchmark
    - name: Run comprehensive test suite
      run: |
        # Tests completos incluyendo performance
        python -m pytest tests/ --cov=src --cov-report=html -v
        python -m pytest -k "performance or stress" --benchmark-json=nightly_benchmark.json
    - name: Archive results
      uses: actions/upload-artifact@v3
      with:
        name: nightly-results
        path: |
          htmlcov/
          nightly_benchmark.json
```

### Weekly Coverage Report
```yaml
# .github/workflows/weekly-report.yml
name: Weekly Coverage Report

on:
  schedule:
    - cron: '0 9 * * 1'  # 9 AM UTC every Monday

jobs:
  coverage-report:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Generate coverage trend
      run: |
        python -m pytest tests/ --cov=src --cov-report=json
        # Script para generar trend report
        python scripts/generate_coverage_trend.py
    - name: Send weekly report
      run: |
        # Enviar reporte por email/slack
        python scripts/send_weekly_report.py
```

## 📊 Metrics Dashboard

### Coverage Tracking
```python
# scripts/track_coverage.py
import json
import datetime

def track_coverage():
    """Track coverage over time"""
    with open('coverage.json') as f:
        coverage_data = json.load(f)
    
    current_coverage = coverage_data['totals']['percent_covered']
    
    # Store in time series database
    metrics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'coverage_percentage': current_coverage,
        'total_lines': coverage_data['totals']['num_statements'],
        'covered_lines': coverage_data['totals']['covered_lines']
    }
    
    # Send to monitoring system
    send_to_influxdb(metrics)
```

### Test Execution Tracking
```python
# scripts/track_test_performance.py
def track_test_performance():
    """Track test execution performance"""
    import subprocess
    import time
    
    start_time = time.time()
    result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                          capture_output=True, text=True)
    execution_time = time.time() - start_time
    
    metrics = {
        'execution_time': execution_time,
        'tests_passed': result.stdout.count('PASSED'),
        'tests_failed': result.stdout.count('FAILED'),
        'tests_skipped': result.stdout.count('SKIPPED')
    }
    
    return metrics
```

## 🎯 Best Practices CI/CD

### 1. Fast Feedback
```yaml
# Ejecutar tests más rápidos primero
jobs:
  quick-tests:
    runs-on: ubuntu-latest
    steps:
    - name: Quick unit tests
      run: python -m pytest tests/unit/ -x  # Para en primer error
  
  slow-tests:
    needs: quick-tests  # Solo si quick-tests pasa
    runs-on: ubuntu-latest
    steps:
    - name: Integration tests
      run: python -m pytest tests/integration/
```

### 2. Parallel Execution
```yaml
# Ejecutar en paralelo cuando sea posible
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10]
    test-group: [unit, integration]
  max-parallel: 6
```

### 3. Conditional Execution
```yaml
# Solo ejecutar tests caros en main branch
- name: Performance tests
  if: github.ref == 'refs/heads/main'
  run: python -m pytest -k "performance"

# Solo ejecutar en cambios relevantes
- name: Strategy tests
  if: contains(github.event.head_commit.message, '[strategy]')
  run: python -m pytest tests/unit/test_strategies/
```

### 4. Artifact Management
```yaml
# Guardar artifacts importantes
- name: Upload test reports
  uses: actions/upload-artifact@v3
  if: always()  # Incluso si tests fallan
  with:
    name: test-reports
    path: |
      tests/reports/
      coverage.xml
      benchmark.json
    retention-days: 30
```

## 🔧 Local CI Simulation

### Pre-commit Hooks
```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: python -m pytest tests/unit/ -x
        language: system
        pass_filenames: false
        always_run: true
```

### Local CI Script
```bash
#!/bin/bash
# scripts/run_ci_locally.sh

echo "🚀 Running local CI simulation..."

# 1. Linting
echo "📝 Running linting..."
black --check src/ tests/ || exit 1
isort --check-only src/ tests/ || exit 1
flake8 src/ tests/ || exit 1

# 2. Unit tests
echo "🧪 Running unit tests..."
python -m pytest tests/unit/ -v || exit 1

# 3. Integration tests
echo "🔗 Running integration tests..."
python -m pytest tests/integration/ -v || exit 1

# 4. Coverage
echo "📊 Running coverage analysis..."
python -m pytest tests/ --cov=src --cov-fail-under=35 || exit 1

echo "✅ All checks passed! Ready for CI/CD"
```

¡CI/CD Pipeline completamente documentado! 🚀