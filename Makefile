.PHONY: install test run lint typecheck setup clean

install:
	pip install -e ".[test]"

test:
	python -m pytest tests/ -v

run:
	python -m memorymesh

setup:
	pip install -e ".[test]"
	python -m memorymesh init
	@echo "---"
	@echo "MemoryMesh ready. Edit .env if needed, then run 'opencode'."

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src/ tests/; \
	else \
		echo "ruff not installed; skipping lint (pip install ruff)"; \
	fi

typecheck:
	@if command -v pyright >/dev/null 2>&1; then \
		pyright src/; \
	else \
		echo "pyright not installed; skipping typecheck (pip install pyright)"; \
	fi

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['db', pathlib.Path('.opencode')/'data', 'build', 'dist']]"
	python -c "import pathlib; [f.unlink(missing_ok=True) for f in pathlib.Path('.').glob('*.log') if f.is_file()]"
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib; [f.unlink(missing_ok=True) for f in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import shutil, pathlib; shutil.rmtree('.pytest_cache', ignore_errors=True)"
	python -c "import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
