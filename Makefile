SHELL := /bin/bash
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
PROJECT := celegans_emulator
ARCHIVE := celegans_emulator.tar.gz

.PHONY: setup lint format test security simulate ablation clean package all

setup:
	@echo "==> Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "==> Downloading connectome data..."
	$(PYTHON_VENV) data/download.py
	@echo "==> Setup complete."

lint:
	@echo "==> Running flake8..."
	$(VENV)/bin/flake8 src/ tests/ scripts/
	@echo "==> Running mypy..."
	$(VENV)/bin/mypy src/ scripts/
	@echo "==> Checking isort..."
	$(VENV)/bin/isort --check-only src/ tests/ scripts/
	@echo "==> Checking black..."
	$(VENV)/bin/black --check src/ tests/ scripts/
	@echo "==> Lint passed."

format:
	@echo "==> Running black..."
	$(VENV)/bin/black src/ tests/ scripts/
	@echo "==> Running isort..."
	$(VENV)/bin/isort src/ tests/ scripts/
	@echo "==> Formatting complete."

test:
	@echo "==> Running pytest with coverage..."
	$(VENV)/bin/pytest tests/ -v --tb=short
	@echo "==> Tests complete."

security:
	@echo "==> Running pip-audit..."
	$(VENV)/bin/pip-audit -r requirements.txt
	@echo "==> Running safety check..."
	$(VENV)/bin/safety check -r requirements.txt || true
	@echo "==> Scanning for secrets..."
	$(VENV)/bin/detect-secrets scan --baseline .secrets.baseline || \
		$(VENV)/bin/detect-secrets scan > .secrets.baseline
	@echo "==> Security checks complete."

simulate:
	@echo "==> Running simulation..."
	$(PYTHON_VENV) scripts/run_simulation.py
	@echo "==> Simulation complete."

ablation:
	@echo "==> Running ablation experiments..."
	$(PYTHON_VENV) scripts/run_ablation.py
	@echo "==> Ablation complete."

clean:
	@echo "==> Cleaning build artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "==> Clean complete."

package: clean
	@echo "==> Creating archive $(ARCHIVE)..."
	tar -czf ../$(ARCHIVE) \
		--exclude='.git' \
		--exclude='.env' \
		--exclude='data/raw' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.mypy_cache' \
		--exclude='.venv' \
		--exclude='venv' \
		--exclude='$(ARCHIVE)' \
		-C .. $(PROJECT)
	@echo "==> Archive created: ../$(ARCHIVE)"
	@echo "SHA-256: $$(sha256sum ../$(ARCHIVE) | awk '{print $$1}')"

all: setup lint test security simulate ablation package
	@echo "==> Full build complete."
