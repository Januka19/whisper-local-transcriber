#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# whisper-local-transcriber :: (Release 3)
#
# Uso:
#   ./run.sh [--rebuild-venv] [--force-install] [--system-deps] [--no-log] [ARGS...]
#
# Flags:
#   --rebuild-venv     Borra y recrea la venv
#   --force-install    Reinstala requirements (force-reinstall)
#   --system-deps      Intenta instalar deps de sistema (requiere sudo)
#   --no-log           No hace tee a archivo de log
#
# Env vars:
#   PYTHON_BIN=python3.12   fuerza python (por ejemplo python3.12)
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

WORK_DIR="work"
OUT_DIR="salida"
LOG_DIR="logs"

REBUILD_VENV=false
FORCE_INSTALL=false
SYSTEM_DEPS=false
GUI=false
NO_LOG_FLAG=false
ARGS=()

say()  { printf "%b\n" "$*"; }
info() { say "ℹ️  $*"; }
ok()   { say "✅ $*"; }
warn() { say "⚠️  $*"; }
die()  { say "❌ $*"; exit 1; }

on_error() {
  local exit_code=$?
  local line_no=${1:-"?"}
  local cmd=${2:-"comando desconocido"}
  say ""
  die "Falló en la línea ${line_no} (exit=${exit_code}). Último comando: ${cmd}"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

need_cmd() { command -v "$1" >/dev/null 2>&1; }

for arg in "$@"; do
  case "$arg" in
    --rebuild-venv) REBUILD_VENV=true ;;
    --force-install) FORCE_INSTALL=true ;;
    --system-deps) SYSTEM_DEPS=true ;;
    --gui) GUI=true ;;
    --no-log) NO_LOG_FLAG=true ;;
    *) ARGS+=("$arg") ;;
  esac
done

# -------------------- selección de Python --------------------
if [[ -z "${PYTHON_BIN:-}" ]]; then
  # Preferimos 3.12 si está disponible; si no, tomamos python3.
  if need_cmd python3.12; then
    PYTHON_BIN="python3.12"
  elif need_cmd python3.11; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

need_cmd "$PYTHON_BIN" || die "No se encontró '$PYTHON_BIN'. Instálalo y vuelve a intentar."

PY_VER="$($PYTHON_BIN -c 'import sys; print("{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))')"
PY_MM="$($PYTHON_BIN -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')"
info "Python seleccionado: $PYTHON_BIN (v$PY_VER)"

if [[ "$PY_MM" =~ ^3\.(13|14|15)$ ]]; then
  warn "Python $PY_MM puede no tener wheels para algunas dependencias en ciertas distros."
  warn "Si tienes problemas instalando, prueba con: PYTHON_BIN=python3.12 ./run.sh --rebuild-venv"
fi

# -------------------- deps de sistema (opcional) --------------------
install_system_deps() {
  # Intento best-effort. Requiere sudo.
  if ! need_cmd sudo; then
    die "--system-deps requiere sudo, pero no está disponible."
  fi

  if need_cmd dnf; then
    info "Detectado dnf (Fedora/RHEL). Instalando ffmpeg y Tk..."
    sudo dnf install -y ffmpeg-free ffmpeg-free-devel python3-tkinter || true
    sudo dnf install -y ffmpeg || true
  elif need_cmd apt-get; then
    info "Detectado apt-get (Debian/Ubuntu). Instalando ffmpeg y Tk..."
    sudo apt-get update -y
    sudo apt-get install -y ffmpeg python3-tk
  else
    warn "No pude detectar gestor de paquetes (dnf/apt-get)."
    warn "Instala manualmente: ffmpeg + ffprobe y (si quieres GUI) tkinter."
  fi
}

if [[ "$SYSTEM_DEPS" == "true" ]]; then
  install_system_deps
fi

# Validar ffmpeg/ffprobe
if ! need_cmd ffmpeg || ! need_cmd ffprobe; then
  warn "No se encontró ffmpeg/ffprobe en PATH."
  warn "- Fedora: sudo dnf install -y ffmpeg-free ffmpeg-free-devel"
  warn "- Ubuntu/Debian: sudo apt-get install -y ffmpeg"
  warn "Puedes reintentar con: ./run.sh --system-deps"
  exit 1
fi

# (GUI removed) tkinter validation no longer needed

# -------------------- venv --------------------
if [[ -d "$VENV_DIR" && "$REBUILD_VENV" == "true" ]]; then
  warn "Se solicitó --rebuild-venv. Eliminando venv..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creando entorno virtual ($VENV_DIR)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Validar venv: revisar pip y python accesibles
python -c "import sys; assert sys.executable" >/dev/null 2>&1 && \
python -m pip --version >/dev/null 2>&1 || {
  warn "Venv dañada o pip no disponible. Recreando..."
  deactivate || true
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  # Verificar pip después de recrear
  python -m pip --version >/dev/null 2>&1 || die "pip sigue no disponible después de recrear venv. Intenta: PYTHON_BIN=python3.12 ./run.sh --rebuild-venv"
}

mkdir -p "$LOG_DIR" "$WORK_DIR" "$OUT_DIR"
RUN_LOG="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"

if [[ "$NO_LOG_FLAG" == "false" ]]; then
  exec > >(tee -a "$RUN_LOG") 2>&1
  info "Log: $RUN_LOG"
fi

info "Actualizando pip/setuptools/wheel..."
python -m pip install -U pip setuptools wheel >/dev/null 2>&1 || warn "No se pudo actualizar pip (continuando)."

# -------------------- deps Python (incluye rich) --------------------
if [[ ! -f "$REQ_FILE" ]]; then
  die "No se encontró $REQ_FILE. No puedo instalar dependencias."
fi

if [[ "$FORCE_INSTALL" == "true" ]]; then
  info "Reinstalando dependencias (--force-install)..."
  python -m pip install --force-reinstall -r "$REQ_FILE"
else
  info "Instalando/verificando dependencias..."
  python -m pip install -r "$REQ_FILE"
fi

# Validaciones: rich y faster_whisper
python -c "import rich; from rich.console import Console" >/dev/null 2>&1 || die "rich no quedó instalado. Revisa pip output arriba."
python -c "import faster_whisper" >/dev/null 2>&1 || die "faster-whisper no quedó instalado. Revisa pip output arriba."
ok "Dependencias OK (rich + faster-whisper)."

# -------------------- entrypoint --------------------
ENTRYPOINT=""
if [[ -f "src/transcriptor.py" ]]; then
  ENTRYPOINT="src/transcriptor.py"
elif [[ -f "transcriptor.py" ]]; then
  ENTRYPOINT="transcriptor.py"
else
  die "No encuentro transcriptor.py (busqué en ./transcriptor.py y ./src/transcriptor.py)."
fi


ok "Entorno listo. Ejecutando transcriptor..."
exec python "$ENTRYPOINT" "${ARGS[@]}"
