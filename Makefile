# RAG Enterprise Assistant - Development Makefile

.PHONY: help install test lint format clean setup dev

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  setup      - Set up development environment"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  clean      - Clean build artifacts"
	@echo "  dev        - Run in development mode"

# Install dependencies
install:
	pip install -r requirements.txt

# Set up development environment
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	@echo "Then run: make install"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Run linting
lint:
	flake8 src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Run in development mode
dev:
	python src/main.py