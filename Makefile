.PHONY: prepare format lint check

prepare:
	uv sync --group dev
	uv run prek install

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

check: format lint
