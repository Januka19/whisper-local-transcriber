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
#   PYTHON_BIN=python3.12   fuerza python si no quieres usar la selección automática
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQ_FILE="requirements.txt"
REQ_STAMP_FILE="$VENV_DIR/.requirements.sha256"
VENV_CREATED=false

WORK_DIR="work"
OUT_DIR="salida"
LOG_DIR="logs"

REBUILD_VENV=false
FORCE_INSTALL=false
SYSTEM_DEPS=false
NO_LOG_FLAG=false
SHOW_RUNNER_HELP=false
ARGS=()
DIARIZATION_ENABLED=true
DIARIZATION_SETUP_REQUESTED=false
DIARIZATION_MODEL_DIR="models/diarization"

say()  { printf "%b\n" "$*"; }
info() { say "ℹ️  $*"; }
ok()   { say "✅ $*"; }
warn() { say "⚠️  $*"; }
die()  { say "❌ $*"; exit 1; }
show_usage() {
  cat <<'EOF'
whisper-local-transcriber :: runner

Uso:
  ./run.sh [flags del runner] [--] [args de transcriptor.py]

Flags del runner:
  -h, --help        Muestra esta ayuda y sale
  --rebuild-venv    Borra y recrea la venv
  --force-install   Reinstala requirements (force-reinstall)
  --system-deps     Intenta instalar deps de sistema (requiere sudo)
  --no-log          No hace tee a archivo de log

Ejemplos:
  ./run.sh
  ./run.sh ruta/al/audio.mp3 --language es
  ./run.sh -- --help
EOF
}

on_error() {
  local exit_code=$?
  local line_no=${1:-"?"}
  local cmd=${2:-"comando desconocido"}
  say ""
  die "Falló en la línea ${line_no} (exit=${exit_code}). Último comando: ${cmd}"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

need_cmd() { command -v "$1" >/dev/null 2>&1; }
python_ok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; }
requirements_fingerprint() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import hashlib
print(hashlib.sha256(Path("requirements.txt").read_bytes()).hexdigest())
PY
}
needs_python_deps_install() {
  local expected_hash current_hash
  expected_hash="$(requirements_fingerprint)"
  if [[ ! -f "$REQ_STAMP_FILE" ]]; then
    return 0
  fi
  current_hash="$(cat "$REQ_STAMP_FILE" 2>/dev/null || true)"
  [[ "$current_hash" != "$expected_hash" ]]
}
write_requirements_stamp() {
  requirements_fingerprint > "$REQ_STAMP_FILE"
}
required_python_deps_available() {
  "$VENV_PYTHON" - <<'PY'
import importlib.util
missing = [name for name in ("faster_whisper", "sherpa_onnx") if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

for arg in "$@"; do
  case "$arg" in
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    -h|--help)
      SHOW_RUNNER_HELP=true
      ;;
    --rebuild-venv) REBUILD_VENV=true ;;
    --force-install) FORCE_INSTALL=true ;;
    --system-deps) SYSTEM_DEPS=true ;;
    --no-log) NO_LOG_FLAG=true ;;
    *) ARGS+=("$arg") ;;
  esac
  shift
done

# Detectar opciones del transcriptor que afectan la preparación automática de
# diarización. La validación completa de argumentos sigue perteneciendo a Python.
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --no-diarize)
      DIARIZATION_ENABLED=false
      ;;
    --diarize)
      DIARIZATION_ENABLED=true
      ;;
    --setup_diarization_models)
      DIARIZATION_SETUP_REQUESTED=true
      ;;
    --diarization_model_dir)
      if ((i + 1 < ${#ARGS[@]})); then
        DIARIZATION_MODEL_DIR="${ARGS[$((i + 1))]}"
        ((i += 1))
      fi
      ;;
    --diarization_model_dir=*)
      DIARIZATION_MODEL_DIR="${ARGS[$i]#*=}"
      ;;
  esac
done

if [[ "$SHOW_RUNNER_HELP" == "true" ]]; then
  show_usage
  exit 0
fi

SKIP_SYSTEM_CHECKS=false
if [[ ${#ARGS[@]} -gt 0 ]]; then
  case "${ARGS[0]}" in
    -h|--help) SKIP_SYSTEM_CHECKS=true ;;
  esac
fi

# -------------------- selección de Python --------------------
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if need_cmd "$candidate" && python_ok "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
need_cmd "$PYTHON_BIN" || die "No se encontró '$PYTHON_BIN'. Instálalo y vuelve a intentar."
python_ok "$PYTHON_BIN" || die "Se requiere Python 3.9 o superior."

PY_VER="$($PYTHON_BIN -c 'import sys; print("{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))')"
info "Python seleccionado: $PYTHON_BIN (v$PY_VER)"

# Ayuda rápida del transcriptor sin preparar entorno completo.
if [[ "$SKIP_SYSTEM_CHECKS" == "true" ]]; then
  if [[ -f "src/transcriptor.py" ]]; then
    exec "$PYTHON_BIN" "src/transcriptor.py" "${ARGS[@]}"
  elif [[ -f "transcriptor.py" ]]; then
    exec "$PYTHON_BIN" "transcriptor.py" "${ARGS[@]}"
  fi
  die "No encuentro transcriptor.py (busqué en ./transcriptor.py y ./src/transcriptor.py)."
fi

# -------------------- deps de sistema (opcional) --------------------
install_system_deps() {
  if ! need_cmd sudo; then
    die "--system-deps requiere sudo, pero no está disponible."
  fi

  if need_cmd dnf; then
    info "Detectado dnf (Fedora/RHEL). Instalando ffmpeg..."
    sudo dnf install -y ffmpeg-free ffmpeg-free-devel || true
    sudo dnf install -y ffmpeg || true
  elif need_cmd apt-get; then
    info "Detectado apt-get (Debian/Ubuntu). Instalando ffmpeg..."
    sudo apt-get update -y
    sudo apt-get install -y ffmpeg
  else
    warn "No pude detectar gestor de paquetes (dnf/apt-get)."
    warn "Instala manualmente: ffmpeg + ffprobe."
  fi
}

if [[ "$SYSTEM_DEPS" == "true" ]]; then
  install_system_deps
fi

if [[ "$SKIP_SYSTEM_CHECKS" != "true" ]]; then
  if ! need_cmd ffmpeg || ! need_cmd ffprobe; then
    warn "No se encontró ffmpeg/ffprobe en PATH."
    warn "- Fedora: sudo dnf install -y ffmpeg-free ffmpeg-free-devel"
    warn "- Ubuntu/Debian: sudo apt-get install -y ffmpeg"
    warn "Puedes reintentar con: ./run.sh --system-deps"
    exit 1
  fi
fi

# -------------------- venv --------------------
if [[ -d "$VENV_DIR" && "$REBUILD_VENV" == "true" ]]; then
  warn "Se solicitó --rebuild-venv. Eliminando venv..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creando entorno virtual ($VENV_DIR)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  VENV_CREATED=true
fi

# Validar venv: revisar pip, python y compatibilidad de versión
"$VENV_PYTHON" -c "import sys; assert sys.executable" >/dev/null 2>&1 && \
"$VENV_PYTHON" -m pip --version >/dev/null 2>&1 && \
python_ok "$VENV_PYTHON" || {
  warn "Venv dañada, desactualizada o pip no disponible. Recreando..."
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  VENV_CREATED=true
  "$VENV_PYTHON" -m pip --version >/dev/null 2>&1 || die "pip sigue no disponible después de recrear venv."
}

mkdir -p "$LOG_DIR" "$WORK_DIR" "$OUT_DIR"
RUN_LOG="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"

if [[ "$NO_LOG_FLAG" == "false" ]]; then
  exec > >(tee -a "$RUN_LOG") 2>&1
  info "Log: $RUN_LOG"
fi

if [[ "$VENV_CREATED" == "true" || "$FORCE_INSTALL" == "true" ]]; then
  info "Actualizando pip/setuptools/wheel..."
  "$VENV_PYTHON" -m pip install -U pip setuptools wheel >/dev/null 2>&1 || warn "No se pudo actualizar pip (continuando)."
fi

# -------------------- deps Python --------------------
if [[ ! -f "$REQ_FILE" ]]; then
  die "No se encontró $REQ_FILE. No puedo instalar dependencias."
fi

if [[ "$FORCE_INSTALL" == "true" || "$REBUILD_VENV" == "true" ]] || needs_python_deps_install || ! required_python_deps_available; then
  if [[ "$FORCE_INSTALL" == "true" ]]; then
    info "Reinstalando dependencias (--force-install)..."
    "$VENV_PYTHON" -m pip install --force-reinstall -r "$REQ_FILE"
  else
    info "Instalando/verificando dependencias..."
    "$VENV_PYTHON" -m pip install -r "$REQ_FILE"
  fi
  write_requirements_stamp
else
  info "Dependencias Python al día; se omite instalación."
fi

required_python_deps_available || die "Dependencias Python incompletas. Reintenta con: ./run.sh --force-install"
ok "Dependencias OK."

# -------------------- entrypoint --------------------
ENTRYPOINT=""
if [[ -f "src/transcriptor.py" ]]; then
  ENTRYPOINT="src/transcriptor.py"
elif [[ -f "transcriptor.py" ]]; then
  ENTRYPOINT="transcriptor.py"
else
  die "No encuentro transcriptor.py (busqué en ./transcriptor.py y ./src/transcriptor.py)."
fi

# La diarización está activa por defecto. En la primera ejecución, preparar sus
# pesos explícitamente antes de entrar al pipeline para que el usuario solo tenga
# que lanzar run.sh. Las siguientes ejecuciones reutilizan los archivos locales.
if [[ "$DIARIZATION_ENABLED" == "true" && "$DIARIZATION_SETUP_REQUESTED" == "false" ]]; then
  if ! "$VENV_PYTHON" -c \
      'import sys; from src.diarization_utils import sherpa_diarization_ready; raise SystemExit(0 if sherpa_diarization_ready(sys.argv[1])[0] else 1)' \
      "$DIARIZATION_MODEL_DIR"; then
    info "Preparando modelos locales de diarización (solo la primera vez)..."
    "$VENV_PYTHON" "$ENTRYPOINT" \
      --setup_diarization_models \
      --diarization_model_dir "$DIARIZATION_MODEL_DIR"
  fi
fi

ok "Entorno listo. Ejecutando transcriptor..."
exec "$VENV_PYTHON" "$ENTRYPOINT" "${ARGS[@]}"
