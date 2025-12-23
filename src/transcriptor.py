#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcriptor Simple v2 (interfaz por consola + diarización ligera + fiabilidad)

Enfoque:
- Mantener un pipeline estable en CPU (sin GPU) para audios largos.
- Añadir diarización "ligera" (etiquetas tipo Participante A/B/...) SIN modelos pesados.
- Mejorar fiabilidad de la interfaz: validaciones, recuperación, logs y ejecución reproducible.

Diarización ligera (importante):
- No es diarización acústica real (no identifica voces por embeddings).
- Es una "segmentación por turnos" basada en pausas/gaps y reglas, más un modo de revisión rápida
  para reasignar etiquetas manualmente si hace falta (muy útil en práctica).

Requisitos:
- ffmpeg + ffprobe en PATH
- pip install -U faster-whisper
"""

import argparse
from argparse import BooleanOptionalAction
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- UI opcional con rich (fallback seguro) ---
RICH_AVAILABLE = False
console = None

# ============================================================================
# INTERFAZ DE USUARIO (con fallback seguro a print)
# ============================================================================

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


def ui_print(msg: str = "") -> None:
    """Imprime con rich si está disponible; caso contrario, print normal."""
    if RICH_AVAILABLE and console is not None:
        console.print(msg)
    else:
        print(msg)


def ui_header() -> None:
    """Encabezado amigable (no afecta la lógica)."""
    if RICH_AVAILABLE and console is not None:
        console.print(
            Panel.fit(
                "[bold cyan]Transcriptor Simple v2[/bold cyan]\n"
                "Offline · CPU-only · audios largos · diarización ligera",
                border_style="cyan",
            )
        )
    else:
        print("\n=== Transcriptor Simple v2 ===\n")

try:
    from faster_whisper import WhisperModel
except Exception:
    ui_print("❌ ERROR: No se pudo importar faster_whisper. Instala con: pip install -U faster-whisper")
    raise SystemExit(1)

# ============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ============================================================================

# Directorios
DEFAULT_WORKDIR = "work"
DEFAULT_OUTDIR = "salida"
DEFAULT_LOGDIR = "logs"

# Modelo y transcripción
DEFAULT_MODEL = "medium"
DEFAULT_FALLBACK_MODEL = "small"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_FALLBACK_COMPUTE_TYPE = "int8"  # Puede ser diferente para ahorrar RAM si falla el principal
DEFAULT_LANGUAGE = "es"

# Chunking de audio
DEFAULT_CHUNK_S = 60
DEFAULT_OVERLAP_S = 0.5
DEFAULT_BEAM = 1
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_NORMALIZE = True

# Post-procesado
DEFAULT_POSTPROCESS = True
DEFAULT_REMOVE_FILLERS = True
DEFAULT_WRITE_CLEAN = True
DEFAULT_MERGE_GAP_S = 0.8

# Deduplicación
DEDUP_WINDOW = 8
SIMILARITY_THRESHOLD = 0.92
MAX_REPEAT_PHRASE = 3

# Diarización ligera (segmentación por turnos)
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_TURN_GAP_S = 1.2
DEFAULT_FORCE_TURN_MAX_S = 30.0
DEFAULT_REVIEW_DIARIZATION = False

# ============================================================================
# TIPOS Y ESTRUCTURAS
# ============================================================================

@dataclass
class Chunk:
    idx: int
    path: str
    start_s: float
    end_s: float

# ============================================================================
# FUNCIONES BÁSICAS (Validación, Ejecución, Sistema de Archivos)
# ============================================================================

def ensure_ffmpeg() -> None:
    """Verifica disponibilidad de ffmpeg y ffprobe en PATH."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        ui_print("❌ No se encontró ffmpeg/ffprobe en PATH. Instala con: sudo apt-get install -y ffmpeg")
        raise SystemExit(1)

def run(cmd: List[str]) -> None:
    """Ejecuta comando shell y levanta excepción si falla."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Comando falló: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def ffprobe_duration_seconds(path: str) -> float:
    """Obtiene duración de audio en segundos usando ffprobe."""
    cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe falló para {path}\n{p.stderr}")
    return float(p.stdout.strip())

def safe_mkdir(path: str) -> None:
    """Crea directorio recursivamente si no existe."""
    os.makedirs(path, exist_ok=True)

def now_stamp() -> str:
    """Retorna timestamp actual en formato YYYYMMDD_HHMMSS."""
    return time.strftime("%Y%m%d_%H%M%S")
def write_log(log_path: str, msg: str) -> None:
    """Escribe log con timestamp a archivo e imprime en consola."""
    safe_mkdir(os.path.dirname(log_path))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    ui_print(msg)

def normalize_to_wav_16k_mono(input_audio: str, out_wav: str) -> None:
    """Normaliza audio a WAV mono 16kHz con filtro pasa-altos y amplitud."""
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-ac", "1",           # Mono
        "-ar", "16000",       # 16 kHz
        "-vn",                # Sin video
        "-af", "highpass=f=80,volume=1.2",  # Filtro + amplificación
        out_wav
    ]
    run(cmd)

def split_audio_fixed(input_audio: str, workdir: str, chunk_s: int, overlap_s: float, prefix: str) -> List[Chunk]:
    """Divide audio en chunks de duración fija con solapamiento."""
    safe_mkdir(workdir)
    dur = ffprobe_duration_seconds(input_audio)
    step = chunk_s - overlap_s
    if step <= 0:
        raise ValueError("chunk_s debe ser mayor que overlap_s.")
    
    chunks: List[Chunk] = []
    idx = 0
    t = 0.0
    
    while t < dur:
        start = max(0.0, t)
        end = min(dur, t + chunk_s)
        duration = end - start
        out_path = os.path.join(workdir, f"{prefix}_chunk_{idx:04d}.wav")
        
        # Extraer chunk con ffmpeg
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", input_audio,
            "-t", str(duration),
            "-ac", "1", "-ar", "16000", "-vn",
            out_path
        ]
        run(cmd)
        chunks.append(Chunk(idx=idx, path=out_path, start_s=start, end_s=end))
        idx += 1
        t += step
    return chunks
# ============================================================================
# FUNCIONES DE UTILIDAD (Rutas, Estado, Archivos)
# ============================================================================

def resolve_audio_path(audio_arg: str) -> str:
    """Resuelve ruta del audio (arrastrar/pegar seguro, relativa, absoluta, con ~)."""
    if not audio_arg:
        raise FileNotFoundError("Ruta de audio vacía.")

    # 🔒 Sanitizar comillas y espacios (caso arrastrar archivo en terminal)
    cleaned = audio_arg.strip().strip("'\"")

    p = Path(cleaned).expanduser().resolve()
    if p.is_file():
        return str(p)

    raise FileNotFoundError(f"No se encontró el audio: {p}")

def load_state(state_path: str) -> Dict[str, Any]:
    """Carga estado guardado o retorna dict vacío."""
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state_path: str, state: Dict[str, Any]) -> None:
    """Guarda estado de forma segura (escribe a tmp primero)."""
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)

# ============================================================================
# FUNCIONES DE PROCESAMIENTO DE TEXTO (Normalización, Similitud, Dedup)
# ============================================================================

def normalize_text_for_similarity(s: str) -> str:
    """Normaliza texto para cálculo de similitud."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sáéíóúñü]", "", s)
    return s

def similarity(a: str, b: str) -> float:
    """Calcula similitud de Jaccard entre dos textos."""
    a = normalize_text_for_similarity(a)
    b = normalize_text_for_similarity(b)
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return inter / union if union else 0.0

def dedup_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Elimina duplicados cercanos permitiendo hasta MAX_REPEAT_PHRASE repeticiones por frase.

    Lógica:
    - Compara el texto actual contra una ventana de anteriores (`DEDUP_WINDOW`) usando `similarity`.
    - Si es similar a alguno en la ventana, incrementa el contador para esa frase y
      permite la repetición solo si aún no alcanza `MAX_REPEAT_PHRASE`.
    - Si no es similar, se considera una nueva frase y su contador se inicializa a 1.
    """
    if not segments:
        return []
    out: List[Dict[str, Any]] = []
    last_texts: List[str] = []
    repeat_counter: Dict[str, int] = {}
    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        key = normalize_text_for_similarity(txt)
        # ¿Existe una frase similar en la ventana reciente?
        is_similar = False
        for prev in last_texts[-DEDUP_WINDOW:]:
            if similarity(txt, prev) >= SIMILARITY_THRESHOLD:
                is_similar = True
                break
        if is_similar:
            # Empezar desde 0 para que la primera repetición sea 1 (más claro)
            repeat_counter[key] = repeat_counter.get(key, 0) + 1
            if repeat_counter[key] <= MAX_REPEAT_PHRASE:
                out.append(s)
                last_texts.append(txt)
            else:
                # superó el máximo permitido → omitir
                continue
        else:
            # nueva frase, inicializar contador
            repeat_counter[key] = 1
            out.append(s)
            last_texts.append(txt)
    return out

DEFAULT_REPLACEMENTS = {r"\bCACA\b":"cacao", r"\bChocoOsama\b":"ChocoSama"}
FILLER_PATTERNS = [
    r"(?i)\b(eee+|mmm+|eh+|em+)\b",
    r"(?i)\b(o sea)\b",
    r"(?i)\b(este|esteee)\b",
    r"(?i)\b(digo)\b",
    r"(?i)\b(ok|okay)\b",
    r"(?i)\b(ya|ajá)\b",
]

# ============================================================================
# FUNCIONES DE LIMPIEZA (Reemplazos, Fillers, Merge)
# ============================================================================

def load_replacements(path: str) -> List[Tuple[re.Pattern, str]]:
    """Carga reemplazos: DEFAULT + usuario (si existe archivo)."""
    reps: List[Tuple[re.Pattern, str]] = [(re.compile(k), v) for k, v in DEFAULT_REPLACEMENTS.items()]
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_map = json.load(f)
        if isinstance(user_map, dict):
            for k, v in user_map.items():
                reps.append((re.compile(str(k)), str(v)))
        elif isinstance(user_map, list):
            for item in user_map:
                if not isinstance(item, dict):
                    continue
                canon = str(item.get("canon","")).strip()
                variantes = item.get("variantes", [])
                if not canon or not isinstance(variantes, list):
                    continue
                for var in variantes:
                    var = str(var).strip()
                    if var:
                        reps.append((re.compile(rf"\b{re.escape(var)}\b"), canon))
    return reps

def clean_text_basic(text: str) -> str:
    """Limpia espacios, normaliza puntuación básica."""
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([¿¡])\s+", r"\1", t)
    return t.strip()

def apply_replacements(text: str, reps: List[Tuple[re.Pattern, str]]) -> str:
    """Aplica reemplazos de regex al texto."""
    t = text
    for pat, repl in reps:
        t = pat.sub(repl, t)
    return t

def remove_fillers(text: str) -> str:
    """Elimina muletillas (umm, eh, o sea, etc)."""
    t = text
    for pat in FILLER_PATTERNS:
        t = re.sub(pat, "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t

def _should_merge(prev_text: str, gap_s: float, merge_gap_s: float) -> bool:
    """Determina si fusionar segmento con el anterior."""
    if gap_s > merge_gap_s:
        return False
    if prev_text.strip().endswith((".", "!", "?", "…")):
        return False
    return True

def postprocess_segments(segments: List[Dict[str, Any]], reps: List[Tuple[re.Pattern, str]], do_remove_fillers: bool, merge_gap_s: float) -> List[Dict[str, Any]]:
    """Post-procesa: limpia, reemplaza, elimina fillers, merge por gaps."""
    cleaned: List[Dict[str, Any]] = []
    for s in segments:
        t = clean_text_basic(s.get("text",""))
        if not t:
            continue
        t = apply_replacements(t, reps)
        if do_remove_fillers:
            t = remove_fillers(t)
        t = clean_text_basic(t)
        if not t:
            continue
        s2 = dict(s); s2["text"] = t
        cleaned.append(s2)
    if not cleaned:
        return []
    merged: List[Dict[str, Any]] = [cleaned[0]]
    for cur in cleaned[1:]:
        prev = merged[-1]
        gap = float(cur["start"]) - float(prev["end"])
        if _should_merge(prev.get("text",""), gap, merge_gap_s):
            prev["text"] = clean_text_basic(prev.get("text","") + " " + cur.get("text",""))
            prev["end"] = cur["end"]
        else:
            merged.append(cur)
    return merged

# ============================================================================
# FUNCIONES DE DIARIZACIÓN (Segmentación por Turnos)
# ============================================================================

def speaker_label(i: int) -> str:
    """Genera etiqueta de participante (A, B, ..., Z, S1, S2, ...)."""
    if 0 <= i < 26:
        return f"Participante {chr(ord('A') + i)}"
    return f"Participante S{i+1}"

def diarize_light(segments: List[Dict[str, Any]], num_speakers: int, turn_gap_s: float, force_turn_max_s: float) -> List[Dict[str, Any]]:
    """Diariza segmentando por turnos basada en pausas y duración."""
    if not segments:
        return []
    num_speakers = max(1, int(num_speakers))
    turn_gap_s = max(0.0, float(turn_gap_s))
    force_turn_max_s = max(1.0, float(force_turn_max_s))
    out = []
    current = 0
    turn_start = float(segments[0]["start"])
    prev_end = float(segments[0]["end"])
    first = dict(segments[0]); first["speaker"] = speaker_label(current); out.append(first)
    for s in segments[1:]:
        start = float(s["start"]); end = float(s["end"])
        gap = start - prev_end
        turn_len = prev_end - turn_start
        change = (gap >= turn_gap_s) or (turn_len >= force_turn_max_s)
        if change:
            current = (current + 1) % num_speakers
            turn_start = start
        s2 = dict(s); s2["speaker"] = speaker_label(current); out.append(s2)
        prev_end = end
    return out

def review_diarization_interactive(segments: List[Dict[str, Any]], num_speakers: int) -> List[Dict[str, Any]]:
    """Revisión interactiva: permite reasignar participantes manualmente.

    Para evitar que el proceso se 'cuelgue' en entornos no interactivos, la función
    detecta si stdin es un TTY y en ese caso omite la revisión (retornando los
    segmentos sin cambios). Además captura EOFError durante la lectura para
    salir de forma segura si la entrada se cierra.
    """
    if not segments:
        return segments
    # No bloquear si no hay terminal interactivo
    try:
        if not sys.stdin or not sys.stdin.isatty():
            ui_print("⚠️ Revisión de diarización solicitada pero no se detectó un terminal interactivo. Se omite la revisión.")
            return segments
    except Exception:
        # En caso de que isatty no esté disponible o falle, evitamos bloquear
        ui_print("⚠️ No puedo validar si el entorno es interactivo; omito la revisión para evitar bloqueo.")
        return segments

    num_speakers = max(1, int(num_speakers))
    max_key = min(num_speakers, 9)
    ui_print("\n=== Revisión de diarización (rápida) ===")
    ui_print(f"Enter=mantener | 1..{max_key}=reasignar | b=atrás | q=salir\n")
    i = 0
    out = [dict(s) for s in segments]
    while i < len(out):
        s = out[i]
        sp = s.get("speaker","Participante A")
        ui_print(f"[{i+1}/{len(out)}] {s['start']:.2f}s -> {s['end']:.2f}s | {sp}")
        ui_print(f"  {s.get('text','')}\n")
        try:
            cmd = input(">> ").strip().lower()
        except EOFError:
            ui_print("\n⚠️ Entrada cerrada (EOF) — saliendo de revisión.")
            break
        if cmd == "":
            i += 1; continue
        if cmd == "q":
            break
        if cmd == "b":
            i = max(0, i-1); continue
        if cmd.isdigit():
            k = int(cmd)
            if 1 <= k <= num_speakers:
                out[i]["speaker"] = speaker_label(k-1)
                i += 1; continue
        ui_print("Comando no válido.\n")
    ui_print("Revisión finalizada.\n")
    return out

def transcribe_one_chunk(model: WhisperModel, chunk_path: str, language: str, beam_size: int, word_timestamps: bool) -> List[Dict[str, Any]]:
    """Transcribe un chunk de audio con el modelo."""
    segments_iter, _ = model.transcribe(chunk_path, language=language, beam_size=beam_size, word_timestamps=word_timestamps)
    out: List[Dict[str, Any]] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            out.append({"start_local": float(seg.start), "end_local": float(seg.end), "text": text})
    return out

def write_outputs(segments_global: List[Dict[str, Any]], meta: Dict[str, Any], out_txt: str, out_json: str) -> None:
    """Escribe transcripción en TXT y JSON."""
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPCIÓN\n============================\n\nMETA\n----------------------------\n")
        for k, v in meta.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")
        for s in segments_global:
            sp = s.get("speaker")
            prefix = f"{sp}: " if sp else ""
            f.write(f"[{s['start']:.2f}s -> {s['end']:.2f}s] {prefix}{s['text']}\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "segments": segments_global}, f, ensure_ascii=False, indent=2)

def clean_workspace(workdir: str, logdir: str, outdir: str, audio_name: str) -> None:
    """
    Borra todo lo intermedio y deja solo las salidas finales.
    NO borra outdir (para conservar *_transcripcion_final.* y *_transcripcion_limpia.*).
    """
    ui_print("\n🧹 Limpieza final (--clean): eliminando intermedios...")

    # Archivos en workdir
    work_files = [
        f"{audio_name}_estado.json",
        f"{audio_name}_partials.jsonl",
        f"{audio_name}_normalized_16k.wav",
        f"{audio_name}_chunks_metadata.json",
]
    for fname in work_files:
        p = Path(workdir) / fname
        if p.exists():
            try:
                p.unlink()
                ui_print(f"  ✔ Eliminado {p}")
            except Exception as e:
                ui_print(f"  ⚠️ No se pudo eliminar {p}: {e}")

    # Carpeta de chunks
    chunks_dir = Path(workdir) / f"{audio_name}_chunks"
    if chunks_dir.exists() and chunks_dir.is_dir():
        try:
            shutil.rmtree(chunks_dir)
            ui_print(f"  ✔ Eliminado {chunks_dir}")
        except Exception as e:
            ui_print(f"  ⚠️ No se pudo eliminar {chunks_dir}: {e}")

    # Logs (solo los de ese audio)
    for p in Path(logdir).glob(f"{audio_name}_*.log"):
        try:
            p.unlink()
            ui_print(f"  ✔ Eliminado {p}")
        except Exception as e:
            ui_print(f"  ⚠️ No se pudo eliminar {p}: {e}")

    ui_print("🧼 Limpieza completada.\n")


def prompt(msg: str, default: Optional[str] = None) -> str:
    if RICH_AVAILABLE and console is not None:
        # Prompt.ask maneja default sin romper nada
        return Prompt.ask(msg, default=default if default is not None else "")
    q = f"{msg} [{default}]: " if default is not None else f"{msg}: "
    s = input(q).strip()
    return s if s else (default if default is not None else "")


def prompt_int(msg: str, default: int, min_v: int = 1, max_v: int = 10000) -> int:
    while True:
        try:
            if RICH_AVAILABLE and console is not None:
                v = IntPrompt.ask(msg, default=default)
            else:
                v = int(prompt(msg, str(default)))

            if v < min_v or v > max_v:
                ui_print(f"  ↳ Ingresa un valor entre {min_v} y {max_v}.")
                continue
            return v
        except Exception:
            ui_print("  ↳ Ingresa un entero válido.")


def prompt_float(msg: str, default: float, min_v: float = 0.0, max_v: float = 10000.0) -> float:
    while True:
        try:
            if RICH_AVAILABLE and console is not None:
                v = FloatPrompt.ask(msg, default=default)
            else:
                v = float(prompt(msg, str(default)))

            if v < min_v or v > max_v:
                ui_print(f"  ↳ Ingresa un valor entre {min_v} y {max_v}.")
                continue
            return v
        except Exception:
            ui_print("  ↳ Ingresa un número válido.")

# ============================================================================
# INTERFAZ INTERACTIVA (Modo Asistido)
# ============================================================================

def assisted_args() -> argparse.Namespace:
    """Modo interactivo: menú, prompts, confirmación antes de ejecutar."""
    ui_header()
    ui_print("\nModo asistido: completa los datos para iniciar.\n")

    # Modo de uso (solo define defaults, no cambia el pipeline)
    if RICH_AVAILABLE and console is not None:
        ui_print("[bold]Selecciona un modo:[/bold]")
        ui_print("  1) Entrevista (más precisión, pausas normales)")
        ui_print("  2) Reunión (balanceado, más robusto)")
        ui_print("  3) Personalizado (tú eliges todo)")
        mode = IntPrompt.ask("Modo", default=2)
    else:
        ui_print("Selecciona un modo:")
        ui_print("  1) Entrevista")
        ui_print("  2) Reunión")
        ui_print("  3) Personalizado")
        mode = int(prompt("Modo", "2") or "2")

    # Defaults sugeridos (ajusta si deseas, pero son seguros)
    if mode == 1:  # Entrevista
        suggested_model = "small"
        suggested_compute = "int8"
        suggested_diarize = True
        suggested_num_speakers = 2
    elif mode == 2:  # Reunión
        suggested_model = "small"
        suggested_compute = "int8"
        suggested_diarize = True
        suggested_num_speakers = 3
    else:  # Personalizado
        suggested_model = DEFAULT_MODEL
        suggested_compute = DEFAULT_COMPUTE_TYPE
        suggested_diarize = DEFAULT_DIARIZE
        suggested_num_speakers = DEFAULT_NUM_SPEAKERS

    while True:
        a = prompt("Ruta del audio (arrastrar/pegar la ruta)", "")
        if not a:
            ui_print("  ↳ Necesito una ruta.")
            continue
        try:
            a = resolve_audio_path(a); break
        except Exception as e:
            ui_print(f"  ↳ {e}")

    model = prompt("Modelo (small/medium o s/m)", suggested_model).lower()
    # Aceptar alias
    if model == "s":
        model = "small"
    elif model == "m":
        model = "medium"
    if model not in {"small","medium"}:
        model = DEFAULT_MODEL

    fallback = prompt("Fallback (small/none o s/n)", DEFAULT_FALLBACK_MODEL).lower()
    # Aceptar alias
    if fallback == "s":
        fallback = "small"
    elif fallback == "n":
        fallback = "none"
    if fallback == "none":
        fallback = ""
    elif fallback not in {"small","medium"}:
        fallback = DEFAULT_FALLBACK_MODEL

    # Compute type para fallback (opcional si tienes modelos limitados en RAM)
    fallback_compute = prompt("Compute para fallback (int8/int16/float16/float32)", DEFAULT_FALLBACK_COMPUTE_TYPE).lower()
    if fallback == "":
        fallback_compute = ""
    elif fallback_compute not in {"int8","int16","float16","float32"}:
        ui_print(f"  ↳ Compute inválido, usando {DEFAULT_FALLBACK_COMPUTE_TYPE} para fallback.")
        fallback_compute = DEFAULT_FALLBACK_COMPUTE_TYPE

    chunk_s = prompt_int("Chunk (seg)", DEFAULT_CHUNK_S, 10, 600)
    overlap_s = prompt_float("Overlap (seg)", DEFAULT_OVERLAP_S, 0.0, float(chunk_s)-0.1)
    beam = prompt_int("Beam (1 recomendado)", DEFAULT_BEAM, 1, 5)

    normalize = prompt("Normalizar a WAV 16k mono (y/n)", "y").lower().startswith("y")
    resume = prompt("Reanudar si existe estado (y/n)", "y").lower().startswith("y")

    post = prompt("Post-procesar (limpieza/nombres) (y/n)", "y").lower().startswith("y")
    remove_fillers = DEFAULT_REMOVE_FILLERS
    merge_gap_s = DEFAULT_MERGE_GAP_S
    replacements_json = ""
    write_clean = DEFAULT_WRITE_CLEAN
    if post:
        remove_fillers = prompt("Quitar muletillas comunes (y/n)", "y").lower().startswith("y")
        merge_gap_s = prompt_float("Fusionar si gap ≤ (s)", DEFAULT_MERGE_GAP_S, 0.0, 10.0)
        replacements_json = prompt("Ruta replacements.json (opcional)", "").strip()
        if replacements_json and not os.path.exists(replacements_json):
            ui_print("  ↳ No encontré ese JSON; continúo sin replacements externos.")
            replacements_json = ""
        write_clean = prompt("Escribir salida limpia adicional (y/n)", "y").lower().startswith("y")

    diarize = prompt("Diarización ligera (Participante A/B/...) (y/n)", "y" if suggested_diarize else "n").lower().startswith("y")
    num_speakers = suggested_num_speakers
    turn_gap_s = DEFAULT_TURN_GAP_S
    force_turn_max_s = DEFAULT_FORCE_TURN_MAX_S
    review_diar = DEFAULT_REVIEW_DIARIZATION
    if diarize:
        num_speakers = prompt_int("Número de participantes", suggested_num_speakers, 1, 9)
        turn_gap_s = prompt_float("Cambio de turno si pausa ≥ (s)", DEFAULT_TURN_GAP_S, 0.1, 10.0)
        force_turn_max_s = prompt_float("Forzar cambio si bloque ≥ (s)", DEFAULT_FORCE_TURN_MAX_S, 5.0, 600.0)
        review_diar = prompt("Revisar diarización al final (y/n)", "n").lower().startswith("y")

    workdir = prompt("Carpeta work", DEFAULT_WORKDIR)
    outdir = prompt("Carpeta salida", DEFAULT_OUTDIR)
    logdir = prompt("Carpeta logs", DEFAULT_LOGDIR)

    # Forzar reutilización de chunks si el usuario lo desea
    force_reuse_chunks = prompt("Forzar reutilización de chunks aunque difieran parámetros (y/n)", "n").lower().startswith("y")

    # Pregunta opcional por idioma (útil en entrevistas mixtas)
    language = prompt("Idioma (ej. 'es' para español, 'en' para inglés)", DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE

    # Crear namespace para el resumen
    args = argparse.Namespace(
        audio=a, workdir=workdir, outdir=outdir, logdir=logdir,
        model=model, fallback_model=fallback, compute_type=suggested_compute, fallback_compute_type=fallback_compute,
        language=language, chunk_s=chunk_s, overlap_s=overlap_s,
        beam=beam, word_timestamps=False, normalize=normalize, resume=resume,
        postprocess=post, replacements_json=replacements_json, merge_gap_s=merge_gap_s,
        remove_fillers=remove_fillers, write_clean=write_clean,
        diarize=diarize, num_speakers=num_speakers, turn_gap_s=turn_gap_s,
        force_turn_max_s=force_turn_max_s, review_diarization=review_diar,
        force_reuse_chunks=force_reuse_chunks, keep_chunks=False
    )

    # Resumen final
    ui_print("\n[bold]Resumen de configuración[/bold]" if RICH_AVAILABLE else "\nResumen de configuración")
    ui_print(f"- Audio: {args.audio}")
    ui_print(f"- Idioma: {args.language}")
    ui_print(f"- Modelo: {args.model} (compute: {args.compute_type})")
    ui_print(f"- Fallback: {args.fallback_model or '(none)'} (compute: {args.fallback_compute_type or '(none)'})")
    ui_print(f"- Diarización: {'Sí' if args.diarize else 'No'}")
    ui_print(f"- Forzar reuse chunks: {'Sí' if args.force_reuse_chunks else 'No'}")
    if args.diarize:
        ui_print(f"- Participantes: {args.num_speakers}")

    if RICH_AVAILABLE and console is not None:
        ok = Confirm.ask("¿Iniciar transcripción?", default=True)
    else:
        ok = (prompt("¿Iniciar transcripción? [S/n]", "S").lower() != "n")

    if not ok:
        ui_print("Ejecución cancelada.")
        raise SystemExit(0)

    return args

def validate_args(args: argparse.Namespace) -> None:
    """Valida argumentos: rangos, tipos, archivos existentes."""
    if args.chunk_s <= 0:
        raise ValueError("chunk_s debe ser > 0")
    if args.overlap_s < 0:
        raise ValueError("overlap_s debe ser >= 0")
    if args.overlap_s >= args.chunk_s:
        raise ValueError("overlap_s debe ser menor que chunk_s")
    if args.beam < 1 or args.beam > 5:
        raise ValueError("beam debe estar entre 1 y 5")
    if args.model not in {"small","medium"}:
        raise ValueError("model debe ser small o medium")
    if args.fallback_model and args.fallback_model not in {"small","medium"}:
        raise ValueError("fallback_model debe ser small/medium/empty")
    if args.compute_type not in {"int8","int16","float16","float32"}:
        raise ValueError("compute_type debe ser int8, int16, float16 o float32")
    # Validar fallback_compute_type solo si se especificó un fallback_model
    if args.fallback_model and args.fallback_model != "":
        if args.fallback_compute_type not in {"int8","int16","float16","float32"}:
            raise ValueError("fallback_compute_type debe ser int8, int16, float16 o float32")
    if args.replacements_json and not os.path.exists(args.replacements_json):
        raise ValueError("replacements_json no existe")
    if args.diarize and (args.num_speakers < 1 or args.num_speakers > 9):
        raise ValueError("num_speakers debe estar entre 1 y 9")

def run_pipeline(args: argparse.Namespace) -> None:
    """Pipeline completo: normaliza, chunking, transcribe, post-procesa, diariza."""
    validate_args(args)
    ensure_ffmpeg()
    safe_mkdir(args.workdir); safe_mkdir(args.outdir); safe_mkdir(args.logdir)

    audio_path = resolve_audio_path(args.audio)
    audio_name = Path(audio_path).stem
    log_path = os.path.join(args.logdir, f"{audio_name}_{now_stamp()}.log")

    state_path = os.path.join(args.workdir, f"{audio_name}_estado.json")
    chunks_dir = os.path.join(args.workdir, f"{audio_name}_chunks")
    normalized_wav = os.path.join(args.workdir, f"{audio_name}_normalized_16k.wav")
    partials_jsonl = os.path.join(args.workdir, f"{audio_name}_partials.jsonl")

    out_txt = os.path.join(args.outdir, f"{audio_name}_transcripcion_final.txt")
    out_json = os.path.join(args.outdir, f"{audio_name}_transcripcion_final.json")

    write_log(log_path, f"📌 Audio: {audio_path}")
    write_log(log_path, f"🧠 Modelo: {args.model} | compute={args.compute_type} | Fallback: {args.fallback_model or '(none)'} (compute={args.fallback_compute_type or '(none)'})")
    write_log(log_path, f"⚙️ chunk={args.chunk_s}s overlap={args.overlap_s}s beam={args.beam} normalize={bool(args.normalize)} resume={bool(args.resume)}")
    write_log(log_path, f"🧼 postprocess={bool(args.postprocess)} remove_fillers={bool(args.remove_fillers)} merge_gap={args.merge_gap_s}s replacements={args.replacements_json or '(default)'}")
    write_log(log_path, f"🗣️ diarize={bool(args.diarize)} num_speakers={getattr(args,'num_speakers',0)} turn_gap={getattr(args,'turn_gap_s',0)} force_turn_max={getattr(args,'force_turn_max_s',0)} review={getattr(args,'review_diarization',False)}")

    state = load_state(state_path) if args.resume else {}
    
    # Validar que el audio y los parámetros críticos sean los mismos si se reanuda
    if state and args.resume:
        saved_audio_path = state.get("audio_path", "")
        saved_compute = state.get("compute_type", "")
        saved_fb_compute = state.get("fallback_compute_type", "")
        saved_chunk_s = state.get("chunk_s")
        saved_overlap = state.get("overlap_s")
        saved_chunks_meta = state.get("chunks_metadata", "")
        mismatch = False
        audio_mismatch = False
        compute_mismatch = False
        fb_compute_mismatch = False
        chunk_s_mismatch = False
        overlap_mismatch = False

        if saved_audio_path != audio_path:
            write_log(log_path, f"❌ ERROR: --resume intenta usar estado de audio diferente")
            write_log(log_path, f"   Guardado: {saved_audio_path}")
            write_log(log_path, f"   Actual:   {audio_path}")
            audio_mismatch = True
        if saved_compute != args.compute_type:
            write_log(log_path, f"❌ ERROR: compute_type difiere del estado guardado (guardado={saved_compute} actual={args.compute_type})")
            compute_mismatch = True
        # Solo comparar fallback compute si ambos tienen fallback model
        if state.get("fallback_model") and args.fallback_model and saved_fb_compute != getattr(args, "fallback_compute_type", ""):
            write_log(log_path, f"❌ ERROR: fallback_compute_type difiere del estado guardado (guardado={saved_fb_compute} actual={getattr(args, 'fallback_compute_type', '')})")
            fb_compute_mismatch = True
        # Comparar parámetros de chunking
        if saved_chunk_s is not None and int(saved_chunk_s) != int(args.chunk_s):
            write_log(log_path, f"❌ ERROR: chunk_s difiere del estado guardado (guardado={saved_chunk_s} actual={args.chunk_s})")
            chunk_s_mismatch = True
        if saved_overlap is not None and float(saved_overlap) != float(args.overlap_s):
            write_log(log_path, f"❌ ERROR: overlap_s difiere del estado guardado (guardado={saved_overlap} actual={args.overlap_s})")
            overlap_mismatch = True

        mismatch = any([audio_mismatch, compute_mismatch, fb_compute_mismatch, chunk_s_mismatch, overlap_mismatch])

        # Permitir forzar reutilización salvo cuando el audio difiere (peligroso)
        if mismatch and getattr(args, "force_reuse_chunks", False):
            if audio_mismatch:
                write_log(log_path, "⚠️ --force_reuse_chunks no permite ignorar diferencia de audio. Reiniciando estado.")
                mismatch = True
            else:
                write_log(log_path, "⚠️ --force_reuse_chunks activo: forzando reuso del estado a pesar de diferencias de parámetros.")
                mismatch = False
        # Si existía metadata de chunks, validar que el input_path coincida con el actual (normalizado o no)
        if saved_chunks_meta:
            try:
                if os.path.exists(saved_chunks_meta):
                    with open(saved_chunks_meta, "r", encoding="utf-8") as f:
                        saved_meta = json.load(f)
                    if saved_meta.get("input_path") != (normalized_wav if args.normalize else audio_path):
                        write_log(log_path, f"❌ ERROR: metadata de chunks fue generada con un audio distinto (guardado={saved_meta.get('input_path')} actual={normalized_wav if args.normalize else audio_path})")
                        mismatch = True
                else:
                    write_log(log_path, f"❌ ERROR: metadata de chunks referenciada no existe: {saved_chunks_meta}")
                    mismatch = True
            except Exception as e:
                write_log(log_path, f"⚠️ Error leyendo metadata de chunks guardada: {e}")
                mismatch = True
        if mismatch:
            write_log(log_path, "   → Reiniciando (se ignora --resume) debido a incompatibilidad de parámetros)")
            # Limpiar artefactos que podrían mezclarse con la nueva ejecución:
            try:
                if os.path.exists(partials_jsonl):
                    os.remove(partials_jsonl)
                    write_log(log_path, f"🧹 Se eliminó archivo parcial: {partials_jsonl}")
            except Exception as e:
                write_log(log_path, f"⚠️ No se pudo eliminar {partials_jsonl}: {e}")
            try:
                # Intentar eliminar el archivo de metadata que fue referenciado en el estado
                # (más seguro que referenciar una variable posiblemente no definida).
                meta_to_remove = saved_chunks_meta or os.path.join(args.workdir, f"{audio_name}_chunks_metadata.json")
                if meta_to_remove and os.path.exists(meta_to_remove):
                    os.remove(meta_to_remove)
                    write_log(log_path, f"🧹 Se eliminó metadata de chunks: {meta_to_remove}")
            except Exception as e:
                write_log(log_path, f"⚠️ No se pudo eliminar metadata de chunks: {e}")
            try:
                for p in Path(chunks_dir).glob(f"{audio_name}_chunk_*.wav"):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                write_log(log_path, f"🧹 Se limpiaron chunks existentes en {chunks_dir}")
            except Exception as e:
                write_log(log_path, f"⚠️ No se pudieron limpiar chunks: {e}")
            state = {}
        else:
            write_log(log_path, f"✓ Reanudando desde estado guardado ({state_path})")
    
    if not state:
        state = {
            "audio_path": audio_path, "audio_name": audio_name,
            "model": args.model, "fallback_model": args.fallback_model,
            "compute_type": args.compute_type, "fallback_compute_type": getattr(args, "fallback_compute_type", ""), "language": args.language,
            "chunk_s": args.chunk_s, "overlap_s": args.overlap_s, "beam": args.beam,
            "normalize": bool(args.normalize),
            "completed_chunks": [], "failed_chunks": [], "total_chunks": None,
            "partials_jsonl": partials_jsonl,
            "chunks_metadata": "",
        }
        save_state(state_path, state)
        write_log(log_path, f"🧾 Estado creado: {state_path}")

    input_for_split = audio_path
    if args.normalize:
        if not os.path.exists(normalized_wav):
            write_log(log_path, "🔧 Normalizando audio a WAV 16k mono...")
            normalize_to_wav_16k_mono(audio_path, normalized_wav)
        input_for_split = normalized_wav

    safe_mkdir(chunks_dir)
    
    # Verificar si podemos reutilizar chunks: deben coincidir parámetros y audio
    chunks_metadata_file = os.path.join(args.workdir, f"{audio_name}_chunks_metadata.json")
    existing = sorted([p for p in Path(chunks_dir).glob(f"{audio_name}_chunk_*.wav")])
    can_reuse = False
    
    if existing and os.path.exists(chunks_metadata_file):
        try:
            with open(chunks_metadata_file, "r", encoding="utf-8") as f:
                meta_chunks = json.load(f)
            # Validar que los parámetros coincidan exactamente
            if (meta_chunks.get("chunk_s") == args.chunk_s and
                meta_chunks.get("overlap_s") == args.overlap_s and
                meta_chunks.get("input_path") == input_for_split):
                can_reuse = True
        except Exception:
            pass

    # Si el usuario forzó reutilización, permitirla aunque los parámetros no coincidan (si existe metadata)
    if not can_reuse and getattr(args, "force_reuse_chunks", False) and existing and os.path.exists(chunks_metadata_file):
        try:
            with open(chunks_metadata_file, "r", encoding="utf-8") as f:
                meta_chunks = json.load(f)
            write_log(log_path, "⚠️ --force_reuse_chunks activo: forzando reutilización de metadata de chunks (usar con cautela).")
            # Registrar al menos los campos clave para facilitar debugging en campo
            try:
                meta_chunk_s = meta_chunks.get("chunk_s")
                meta_overlap_s = meta_chunks.get("overlap_s")
                meta_input_path = meta_chunks.get("input_path")
                write_log(log_path, f"   metadata: chunk_s={meta_chunk_s} overlap_s={meta_overlap_s} input_path={meta_input_path}")
                write_log(log_path, f"   current:  chunk_s={args.chunk_s} overlap_s={args.overlap_s} input_path={input_for_split}")
                diffs = []
                if meta_chunk_s != args.chunk_s:
                    diffs.append("chunk_s")
                if meta_overlap_s != args.overlap_s:
                    diffs.append("overlap_s")
                if meta_input_path != input_for_split:
                    diffs.append("input_path")
                write_log(log_path, f"   campos que difieren: {', '.join(diffs) if diffs else 'ninguno'}")
            except Exception:
                # No queremos que el logging de diagnóstico impida el reuso
                pass
            can_reuse = True
        except Exception as e:
            write_log(log_path, f"⚠️ No pude cargar metadata de chunks para forzar reuso: {e}")
            can_reuse = False

    if can_reuse:
        # Reutilizar: cargar metadata que tiene los timecodes correctos
        try:
            chunks: List[Chunk] = []
            for item in meta_chunks.get("chunks", []):
                chunks.append(Chunk(
                    idx=int(item["idx"]),
                    path=item["path"],
                    start_s=float(item["start_s"]),
                    end_s=float(item["end_s"])
                ))
            chunks.sort(key=lambda c: c.idx)
            write_log(log_path, f"♻️ Reutilizando {len(chunks)} chunks existentes (parámetros coinciden).")
            # Asegurar que el estado apunte al archivo de metadata
            try:
                state["chunks_metadata"] = chunks_metadata_file
                save_state(state_path, state)
            except Exception:
                pass
        except Exception as e:
            write_log(log_path, f"⚠️ Metadatos de chunks corruptos: {e}. Regenerando...")
            can_reuse = False
    
    if not can_reuse:
        write_log(log_path, "✂️ Generando chunks...")
        chunks = split_audio_fixed(input_for_split, chunks_dir, args.chunk_s, args.overlap_s, audio_name)
        write_log(log_path, f"✅ Chunks generados: {len(chunks)}")
        
        # Guardar metadata para reutilización futura
        meta_chunks_obj = {
            "chunk_s": args.chunk_s,
            "overlap_s": args.overlap_s,
            "input_path": input_for_split,
            "chunks": [
                {"idx": ch.idx, "path": ch.path, "start_s": ch.start_s, "end_s": ch.end_s}
                for ch in chunks
            ]
        }
        with open(chunks_metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_chunks_obj, f, ensure_ascii=False, indent=2)
        write_log(log_path, f"💾 Metadata de chunks guardada: {chunks_metadata_file}")
        # Guardar ruta de metadata en el estado para validación futura
        try:
            state["chunks_metadata"] = chunks_metadata_file
            save_state(state_path, state)
        except Exception:
            pass

    state["total_chunks"] = len(chunks)
    save_state(state_path, state)

    write_log(log_path, f"🧠 Cargando modelo principal: {args.model} ({args.compute_type})")
    model_main = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)

    model_fallback = None
    if args.fallback_model and args.fallback_model != args.model:
        write_log(log_path, f"🧠 Cargando fallback: {args.fallback_model} ({args.fallback_compute_type})")
        model_fallback = WhisperModel(args.fallback_model, device="cpu", compute_type=args.fallback_compute_type)

    completed = set(state.get("completed_chunks", []))
    failed = set(state.get("failed_chunks", []))
    if not os.path.exists(partials_jsonl):
        open(partials_jsonl, "w", encoding="utf-8").close()

    t0 = time.time()

    # Usar rich.Progress para mostrar progreso si está disponible (solo presentación)
    use_progress = RICH_AVAILABLE and console is not None
    progress = None
    task_chunks = None
    task_segments = None
    segments_count_initial = 0
    segments_done = 0

    # Contar segmentos ya procesados (si reanudamos desde partials_jsonl)
    if os.path.exists(partials_jsonl):
        try:
            with open(partials_jsonl, "r", encoding="utf-8") as f:
                segments_count_initial = sum(1 for line in f if line.strip())
        except Exception:
            segments_count_initial = 0
    segments_done = segments_count_initial

    if use_progress:
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} chunks"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            progress.__enter__()
            task_chunks = progress.add_task("Chunks", total=len(chunks))
            # Crear task de segmentos con total dinámico y valor inicial
            task_segments = progress.add_task("Segmentos", total=max(segments_count_initial, 0))
            if segments_count_initial > 0:
                try:
                    progress.update(task_segments, completed=segments_count_initial)
                except Exception:
                    pass
        except Exception:
            progress = None
            task_chunks = None
            task_segments = None

    try:
        for ch in chunks:
            # Actualizar descripción del task (si aplica)
            if use_progress and task_chunks is not None:
                try:
                    progress.update(task_chunks, description=f"Chunk {ch.idx+1}/{len(chunks)}")
                except Exception:
                    pass

            # Envolvemos la iteración en try/finally para asegurar advance() siempre
            try:
                if ch.idx in completed:
                    # Ya procesado: avanzar y continuar
                    write_log(log_path, f"↪️ Saltando chunk {ch.idx} (ya completado)")
                    continue

                write_log(log_path, f"\n▶️ Chunk {ch.idx+1}/{len(chunks)}  [{ch.start_s/60:.1f}m -> {ch.end_s/60:.1f}m]")

                def persist_failure(err: str) -> None:
                    failed.add(ch.idx)
                    state["failed_chunks"] = sorted(list(failed))
                    save_state(state_path, state)
                    write_log(log_path, f"⚠️ Fallido: chunk {ch.idx} | {err}")

                try:
                    local_segments = transcribe_one_chunk(model_main, ch.path, args.language, args.beam, bool(args.word_timestamps))
                except Exception as e:
                    write_log(log_path, f"⚠️ Falló con principal: {e}")
                    if model_fallback is None:
                        persist_failure(str(e))
                        continue
                    try:
                        write_log(log_path, "🔁 Reintentando con fallback...")
                        local_segments = transcribe_one_chunk(model_fallback, ch.path, args.language, args.beam, bool(args.word_timestamps))
                    except Exception as e2:
                        persist_failure(str(e2))
                        continue

                segs_global: List[Dict[str, Any]] = []
                for s in local_segments:
                    segs_global.append({
                        "chunk_idx": ch.idx,
                        "start": float(ch.start_s + s["start_local"]),
                        "end": float(ch.start_s + s["end_local"]),
                        "text": s["text"],
                    })

                with open(partials_jsonl, "a", encoding="utf-8") as f:
                    for sg in segs_global:
                        f.write(json.dumps(sg, ensure_ascii=False) + "\n")

                completed.add(ch.idx)
                state["completed_chunks"] = sorted(list(completed))
                save_state(state_path, state)
                write_log(log_path, f"✅ Chunk {ch.idx} OK ({len(segs_global)} segs)")

                # Avanzar contador de segmentos en la barra (presentación) y mostrar seg/s
                if use_progress and task_segments is not None and segs_global:
                    try:
                        segments_done += len(segs_global)
                        # Asegurar total >= completed
                        try:
                            cur_total = progress.tasks[task_segments].total or 0
                        except Exception:
                            cur_total = 0
                        if segments_done > cur_total:
                            progress.update(task_segments, total=segments_done)
                        progress.update(task_segments, completed=segments_done)
                        elapsed = max(1e-6, time.time() - t0)
                        segs_per_s = segments_done / elapsed
                        progress.update(task_segments, description=f"{segments_done} segs ({segs_per_s:.2f} seg/s)")
                    except Exception:
                        pass

            finally:
                if use_progress and task_chunks is not None:
                    try:
                        progress.advance(task_chunks)
                    except Exception:
                        pass
    finally:
        if progress is not None:
            try:
                progress.__exit__(None, None, None)
            except Exception:
                pass

    segments_global: List[Dict[str, Any]] = []
    with open(partials_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                segments_global.append(json.loads(line))
            except Exception:
                continue

    segments_global.sort(key=lambda x: (x["start"], x["end"]))
    segments_global = dedup_segments(segments_global)

    meta = {
        "audio": audio_name, "audio_path": audio_path,
        "model": args.model, "fallback_model": args.fallback_model,
        "compute_type": args.compute_type, "fallback_compute_type": getattr(args, "fallback_compute_type", ""), "language": args.language,
        "chunk_s": args.chunk_s, "overlap_s": args.overlap_s,
        "beam": args.beam, "normalized": bool(args.normalize),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_total": len(chunks), "chunks_ok": len(completed), "chunks_failed": len(failed),
        "log": log_path,
    }

    segments_clean = segments_global
    if args.postprocess:
        reps = load_replacements(args.replacements_json)
        segments_clean = postprocess_segments(segments_global, reps, bool(args.remove_fillers), float(args.merge_gap_s))

    segments_for_diar = segments_clean if args.postprocess else segments_global
    if args.diarize:
        diar = diarize_light(segments_for_diar, int(args.num_speakers), float(args.turn_gap_s), float(args.force_turn_max_s))
        if getattr(args, "review_diarization", False):
            diar = review_diarization_interactive(diar, int(args.num_speakers))
        segments_for_write = diar
        meta["diarization"] = {
            "type": "light_rules",
            "num_speakers": int(args.num_speakers),
            "turn_gap_s": float(args.turn_gap_s),
            "force_turn_max_s": float(args.force_turn_max_s),
            "reviewed": bool(getattr(args, "review_diarization", False)),
        }
    else:
        segments_for_write = segments_for_diar

    write_outputs(segments_for_write, meta, out_txt, out_json)

    if args.write_clean and args.postprocess:
        out_txt_clean = os.path.join(args.outdir, f"{audio_name}_transcripcion_limpia.txt")
        out_json_clean = os.path.join(args.outdir, f"{audio_name}_transcripcion_limpia.json")
        meta_clean = dict(meta); meta_clean["note"] = "Salida limpia (sin diarización), lista para análisis."
        write_outputs(segments_clean, meta_clean, out_txt_clean, out_json_clean)
        write_log(log_path, f"🧼 Limpia: {out_txt_clean} y {out_json_clean}")

    t1 = time.time()
    write_log(log_path, "\n✅ Listo.")
    write_log(log_path, f"TXT:  {out_txt}")
    write_log(log_path, f"JSON: {out_json}")
    if failed:
        write_log(log_path, f"⚠️ Chunks fallidos: {sorted(list(failed))}")
    write_log(log_path, f"Tiempo total: {(t1-t0)/60:.1f} min\n")

if getattr(args, "clean", False):
    clean_workspace(
        workdir=args.workdir,
        logdir=args.logdir,
        outdir=args.outdir,
        audio_name=audio_name
    )

# ============================================================================
# CLI Y PUNTO DE ENTRADA
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Construye parser de línea de comandos."""
    p = argparse.ArgumentParser(description="Transcriptor Simple v2 (modo asistido si no pasas audio).")
    p.add_argument("audio", nargs="?", default=None, help="Ruta del audio. Si omites, entra en asistente.")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--logdir", default=DEFAULT_LOGDIR)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["small","medium"])
    p.add_argument("--fallback_model", default=DEFAULT_FALLBACK_MODEL, choices=["","small","medium"])
    p.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE, choices=["int8","int16","float16","float32"], help="Tipo de precisión para el modelo principal")
    p.add_argument("--fallback_compute_type", default=DEFAULT_FALLBACK_COMPUTE_TYPE, choices=["int8","int16","float16","float32"], help="Tipo de precisión para el modelo fallback")
    p.add_argument("--language", default=DEFAULT_LANGUAGE)
    p.add_argument("--chunk_s", type=int, default=DEFAULT_CHUNK_S)
    p.add_argument("--overlap_s", type=float, default=DEFAULT_OVERLAP_S)
    p.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    p.add_argument("--word_timestamps", action=BooleanOptionalAction, default=DEFAULT_WORD_TIMESTAMPS)
    p.add_argument("--normalize", action=BooleanOptionalAction, default=DEFAULT_NORMALIZE)
    p.add_argument("--resume", action=BooleanOptionalAction, default=True)

    p.add_argument("--postprocess", action=BooleanOptionalAction, default=DEFAULT_POSTPROCESS)
    p.add_argument("--replacements_json", default="")
    p.add_argument("--merge_gap_s", type=float, default=DEFAULT_MERGE_GAP_S)
    p.add_argument("--remove_fillers", action=BooleanOptionalAction, default=DEFAULT_REMOVE_FILLERS)
    p.add_argument("--write_clean", action=BooleanOptionalAction, default=DEFAULT_WRITE_CLEAN)

    p.add_argument("--diarize", action=BooleanOptionalAction, default=DEFAULT_DIARIZE)
    p.add_argument("--num_speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    p.add_argument("--turn_gap_s", type=float, default=DEFAULT_TURN_GAP_S)
    p.add_argument("--force_turn_max_s", type=float, default=DEFAULT_FORCE_TURN_MAX_S)
    p.add_argument("--review_diarization", action=BooleanOptionalAction, default=DEFAULT_REVIEW_DIARIZATION)
    p.add_argument("--force_reuse_chunks", action=BooleanOptionalAction, default=False, help="Forzar reutilización de chunks aunque difieran parámetros (usar con cautela)")
    p.add_argument("--clean", action=BooleanOptionalAction, default=False, help="Eliminar archivos intermedios y dejar solo salidas finales")
    return p

def main() -> None:
    """Punto de entrada: CLI → asistido o pipeline."""
    args = build_parser().parse_args()
    if args.audio is None:
        args = assisted_args()  # Modo interactivo si no se pasa audio
    run_pipeline(args)


if __name__ == "__main__":
    main()
