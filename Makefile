.DEFAULT_GOAL := help

.PHONY: install lint format format-check typecheck test deptry check help

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

check: lint format-check typecheck test deptry ## Run all checks
