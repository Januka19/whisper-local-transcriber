#!/usr/bin/env bash
set -e

# Verifica venv
if [ ! -d ".venv" ]; then
  echo "❌ No existe el entorno virtual (.venv)."
  echo "   Ejecuta primero:"
  echo "   python3 -m venv .venv"
  echo "   source .venv/bin/activate"
  echo "   pip install -r requirements.txt"
  exit 1
fi

# Activa venv
source .venv/bin/activate

# Ejecuta el transcriptor
exec python src/transcriptor.py "$@"
