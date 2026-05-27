# ==============================================================================
# MemoryMesh v0.5.0 Makefile
# ==============================================================================

.PHONY: help install install-local install-cli install-all test test-unit test-int run sessions stats setup lint typecheck clean

# Default target
help:
	@echo "🧠 MemoryMesh Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Install basic package with test dependencies (remote embedding)"
	@echo "  make install-local - Install with local embedding models support"
	@echo "  make install-cli   - Install with CLI rich tables support"
	@echo "  make install-all   - Install with ALL dependencies (Recommended for Dev)"
	@echo ""
	@echo "Development & Testing:"
	@echo "  make test          - Run all test suites"
	@echo "  make test-unit     - Run only unit tests"
	@echo "  make test-int      - Run only integration tests"
	@echo "  make lint          - Run ruff linter"
	@echo "  make typecheck     - Run pyright type checker"
	@echo ""
	@echo "Execution & CLI:"
	@echo "  make run           - Run the MemoryMesh server manually"
	@echo "  make sessions      - List recent sessions via CLI"
	@echo "  make stats         - Show system statistics via CLI"
	@echo "  make setup         - Full install + workspace initialization"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Remove databases, cache, logs, and build artifacts"

# ==============================================================================
# Installation
# ==============================================================================
install:
	@echo "📦 Installing MemoryMesh (Basic)..."
	pip install -e ".[test]"

install-local:
	@echo "📦 Installing MemoryMesh (Local Embeddings)..."
	pip install -e ".[test,local]"

install-cli:
	@echo "📦 Installing MemoryMesh (CLI Tools)..."
	pip install -e ".[test,cli]"

install-all:
	@echo "📦 Installing MemoryMesh (All dependencies)..."
	pip install -e ".[test,local,cli]"

# ==============================================================================
# Testing
# ==============================================================================
test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v

test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v

test-int:
	@echo "🧪 Running integration tests..."
	python -m pytest tests/integration/ -v

# ==============================================================================
# Execution & CLI
# ==============================================================================
run:
	@echo "🚀 Starting MemoryMesh server..."
	python -m memorymesh

sessions:
	@echo "📋 Fetching recent sessions..."
	python -m memorymesh sessions

stats:
	@echo "📊 Fetching system statistics..."
	python -m memorymesh stats

setup:
	@echo "⚙️  Setting up MemoryMesh workspace..."
	pip install -e ".[test,local,cli]"
	python -m memorymesh init
	@echo "---"
	@echo "✅ MemoryMesh is ready! Edit the .env file if needed, then run your AI agent (e.g., 'opencode')."

# ==============================================================================
# Development
# ==============================================================================
lint:
	@echo "🧹 Running ruff linter..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src/ tests/; \
	else \
		echo "⚠️ ruff not installed; skipping lint (pip install ruff)"; \
	fi

typecheck:
	@echo "🔎 Running pyright type checker..."
	@if command -v pyright >/dev/null 2>&1; then \
		pyright src/; \
	else \
		echo "⚠️ pyright not installed; skipping typecheck (pip install pyright)"; \
	fi

# ==============================================================================
# Maintenance
# ==============================================================================
clean:
	@echo "🗑️  Cleaning up workspace (caches, logs, and databases)..."
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['db', 'build', 'dist']]"
	python -c "import pathlib; [f.unlink(missing_ok=True) for f in pathlib.Path('.').glob('*.log') if f.is_file()]"
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib; [f.unlink(missing_ok=True) for f in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import shutil, pathlib; shutil.rmtree('.pytest_cache', ignore_errors=True)"
	python -c "import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
	@echo "✅ Cleanup complete."