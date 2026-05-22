#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " MemoryMesh Setup"
echo "============================================"

# --- Detect OS ---
case "$(uname -s)" in
    Linux*)   OS=linux ;;
    Darwin*)  OS=macos ;;
    *)        echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac
echo "[1/4] Platform: $OS"

# --- Create .env if missing ---
if [ ! -f .env ]; then
    echo "[2/4] Creating .env from .env.example..."
    cp .env.example .env
    echo "      -> Edit .env to set your LLM endpoint and model."
else
    echo "[2/4] .env already exists, skipping."
fi

# --- Create virtual environment ---
if [ ! -d .venv ]; then
    echo "[3/4] Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "[3/4] Virtual environment already exists, skipping."
fi

# --- Activate & install ---
echo "[4/4] Installing MemoryMesh..."
source .venv/bin/activate
pip install -e ".[test]"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Start:     python -m memorymesh"
echo "  Test:      python -m pytest tests/ -v"
echo ""
