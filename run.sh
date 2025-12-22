#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# whisper-local-transcriber :: robust runner
#
# Features:
# - crea venv si no existe
# - valida venv existente (y la recrea si se pide)
# - instala dependencias (forzado u automático)
# - valida ffmpeg / ffprobe
# - crea carpetas runtime
# - flags de control:
#     --rebuild-venv     borra y recrea la venv
#     --force-install    reinstala requirements
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="requirements.txt"
ENTRYPOINT="src/transcriptor.py"

WORK_DIR="work"
OUT_DIR="salida"
LOG_DIR="logs"

REBUILD_VENV=false
FORCE_INSTALL=false
ARGS=()

# ------------------ helpers ------------------
say()  { printf "%b\n" "$*"; }
die()  { say "❌ $*"; exit 1; }
warn() { say "⚠️  $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "No se encontró '$1'. Instálalo y vuelve a intentar."
}

# ------------------ parse flags ------------------
for arg in "$@"; do
  case "$arg" in
    --rebuild-venv) REBUILD_VENV=true ;;
    --force-install) FORCE_INSTALL=true ;;
    *) ARGS+=("$arg") ;;
  esac
done

# ------------------ checks base ------------------
need_cmd "$PYTHON_BIN"
need_cmd ffmpeg
need_cmd ffprobe

# ------------------ venv handling ------------------
if [ -d "$VENV_DIR" ] && $REBUILD_VENV; then
  warn "Se solicitó --rebuild-venv. Eliminando entorno virtual existente..."
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  say "ℹ️  Creando entorno virtual ($VENV_DIR)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR" || die "No se pudo crear el entorno virtual."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ------------------ validate venv integrity ------------------
if ! python -c "import sys" >/dev/null 2>&1; then
  warn "El entorno virtual parece estar dañado. Recreando..."
  deactivate || true
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

# ------------------ pip & deps ------------------
python -m pip install --upgrade pip >/dev/null 2>&1 || warn "No se pudo actualizar pip."

if [ ! -f "$REQ_FILE" ]; then
  warn "No se encontró $REQ_FILE. Saltando instalación de dependencias."
else
  if $FORCE_INSTALL; then
    say "ℹ️  Reinstalando dependencias (--force-install)..."
    pip install --force-reinstall -r "$REQ_FILE" || die "Falló la reinstalación."
  else
    say "ℹ️  Verificando dependencias..."
    pip install -r "$REQ_FILE" || die "Falló la instalación de dependencias."
  fi
fi

# ------------------ runtime dirs ------------------
mkdir -p "$WORK_DIR" "$OUT_DIR" "$LOG_DIR" || die "No se pudieron crear carpetas runtime."

# ------------------ entrypoint ------------------
[ -f "$ENTRYPOINT" ] || die "No se encuentra el script principal: $ENTRYPOINT"

# ------------------ run ------------------
say "✅ Entorno listo. Ejecutando transcriptor..."
exec python "$ENTRYPOINT" "${ARGS[@]}"