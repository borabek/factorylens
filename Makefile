.PHONY: install lint typecheck test check paper

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest -q --cov

check: lint typecheck test

# needs tectonic (brew install tectonic); tables come from results/, never typed by hand
paper:
	uv run python scripts/paper_assets.py
	cd paper && tectonic -X compile main.tex
