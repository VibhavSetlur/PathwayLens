# PathwayLens 2.0 Makefile

.PHONY: help install install-dev test test-unit test-integration test-e2e test-coverage lint format type-check clean build docker-build docker-run docs

# Default target
help:
	@echo "PathwayLens 2.0 Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Installation:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-e2e     Run end-to-end tests only"
	@echo "  test-coverage  Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Development:"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  build        Build the project"
	@echo "  docs         Generate documentation"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker images"
	@echo "  docker-run   Run Docker containers"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	python3 scripts/tests/run_tests.py --test-type all

test-unit:
	python3 scripts/tests/run_tests.py --test-type unit

test-integration:
	python3 scripts/tests/run_tests.py --test-type integration

test-e2e:
	python3 scripts/tests/run_tests.py --test-type e2e

test-coverage:
	python3 scripts/tests/run_tests.py --test-type coverage

test-system:
	python3 scripts/tests/test_system.py

# Code Quality
lint:
	python3 scripts/tests/run_tests.py --lint

format:
	black pathwaylens_core/ pathwaylens_api/ pathwaylens_cli/
	isort pathwaylens_core/ pathwaylens_api/ pathwaylens_cli/

type-check:
	python3 scripts/tests/run_tests.py --type-check

# Development
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .tox/

build:
	python -m build

docs:
	cd docs && make html

# Docker
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

# CI/CD
ci-test:
	python3 scripts/tests/run_tests.py --full

# Database
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# API
api-start:
	uvicorn pathwaylens_api.main:app --reload --host 0.0.0.0 --port 8000

api-test:
	curl -X GET "http://localhost:8000/health" -H "accept: application/json"

# CLI
cli-test:
	pathwaylens --help

# Frontend
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-test:
	cd frontend && npm test

# Plugin Development
plugin-create:
	@echo "Creating new plugin template..."
	@read -p "Enter plugin name: " name; \
	mkdir -p pathwaylens_core/plugins/$$name; \
	cp pathwaylens_core/plugins/example_plugin.py pathwaylens_core/plugins/$$name/__init__.py

# Monitoring
monitor-start:
	docker-compose -f docker-compose.monitoring.yml up -d

monitor-stop:
	docker-compose -f docker-compose.monitoring.yml down

# Backup
backup:
	@echo "Creating backup..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "pathwaylens_backup_$$timestamp.tar.gz" \
		--exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.pytest_cache' \
		--exclude='htmlcov' \
		--exclude='.coverage' \
		--exclude='node_modules' \
		--exclude='.next' \
		--exclude='dist' \
		--exclude='build' \
		.

# Release
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major

# Security
security-scan:
	safety check
	bandit -r pathwaylens_core/ pathwaylens_api/ pathwaylens_cli/

# Performance
perf-test:
	python -m pytest tests/performance/ -v

# Load Testing
load-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8000
