# ---------------------------------------------------------------
# Makefile: short names for the main project commands.
# Pipeline targets (data, features, backtest, report, all) are
# added in the phases that create the corresponding scripts.
# ---------------------------------------------------------------

.PHONY: install lint format test check

# Creates the virtual environment and installs the locked dependencies.
install:
	uv sync

# Reports style problems and likely errors without changing any file.
lint:
	uv run ruff check .
	uv run ruff format --check .

# Rewrites files to the project style and applies safe automatic fixes.
format:
	uv run ruff format .
	uv run ruff check --fix .

# Runs the unit tests.
test:
	uv run pytest

# Runs the same checks as the CI workflow.
check: lint test
