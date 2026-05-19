.DEFAULT_GOAL := help

.PHONY: install lint format format-check typecheck test deptry check bench bench-compare help

install: ## Install dependencies (dev group)
	uv sync --group dev

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

format-check: ## Check formatting without modifying files
	ruff format --check src/ tests/

typecheck: ## Run mypy type checker
	mypy

test: ## Run test suite
	pytest tests/ -v

deptry: ## Check for dependency issues
	deptry src/

bench: ## Run micro-benchmarks
	PYTHONHASHSEED=0 uv run pytest benchmarks/ -c benchmarks/pytest.ini
	uv run python benchmarks/report.py benchmarks/benchmark.json

bench-compare: ## Compare against saved baseline (fail >10% regression)
	PYTHONHASHSEED=0 uv run pytest benchmarks/ -c benchmarks/pytest.ini --benchmark-compare --benchmark-compare-fail=10%

check: lint format-check typecheck test deptry ## Run all checks
