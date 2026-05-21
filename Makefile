.PHONY: install test run lint typecheck clean

install:
	pip install -e .

test:
	python -m pytest tests/ -v

run:
	python -m memorymesh

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src/ tests/; \
	else \
		echo "ruff not installed; skipping lint"; \
	fi

typecheck:
	@if command -v pyright >/dev/null 2>&1; then \
		pyright src/; \
	else \
		echo "pyright not installed; skipping typecheck"; \
	fi

clean:
	rm -rf db/ .opencode/data/ *.log .pytest_cache/ build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
