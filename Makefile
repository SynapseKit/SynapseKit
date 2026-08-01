.DEFAULT_GOAL := help

.PHONY: install lint format format-check typecheck test deptry check bench bench-compare release-check release-check-live help

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
	PYTHONHASHSEED=0 uv run pytest benchmarks/ -c benchmarks/pytest.ini --benchmark-compare --benchmark-compare-fail=mean:10%

release-check: ## Run the release-validation harness (offline, no API keys)
	uv run python -m release_check --md release_check_report.md

release-check-live: ## Run the release-validation harness incl. live LLM checks (needs API keys)
	uv run python -m release_check --live --md release_check_report.md

check: lint format-check typecheck test deptry ## Run all checks
