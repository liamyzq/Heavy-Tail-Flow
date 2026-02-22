#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv not found. Install with: python -m pip install --user uv"
  exit 1
fi

echo "Using uv: $UV_BIN"
"$UV_BIN" --version

if [ ! -f "pyproject.toml" ]; then
  echo "pyproject.toml not found in $ROOT_DIR"
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Using existing virtual environment at $ROOT_DIR/.venv"
else
  "$UV_BIN" venv
fi

"$UV_BIN" sync

echo
echo "Setup complete."
echo "Activate with: source $ROOT_DIR/.venv/bin/activate"
