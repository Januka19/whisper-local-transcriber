#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
whisper-local-transcriber :: Transcriptor Release 3 (estable)

Objetivo:
- Offline, CPU-friendly, audios largos, ejecución reproducible.
- Resume real (estado + parciales), con validación de compatibilidad.
- Postproceso opcional + diarización ligera por reglas (turnos por pausas).
- Modelos flexibles: acepta IDs de HuggingFace / rutas locales / alias.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.audio_utils import Chunk, normalize_to_wav_16k_mono, split_audio_fixed
    from src.diarization_utils import diarize_light, review_diarization_interactive
    from src.state_utils import config_signature, load_state, resolve_audio_path, save_state
    from src.system_utils import (
        Logger,
        audio_needs_normalization,
        ensure_ffmpeg,
        now_stamp,
        probe_audio_info,
        safe_mkdir,
    )
except ModuleNotFoundError:
    from audio_utils import Chunk, normalize_to_wav_16k_mono, split_audio_fixed
    from diarization_utils import diarize_light, review_diarization_interactive
    from state_utils import config_signature, load_state, resolve_audio_path, save_state
    from system_utils import (
        Logger,
        audio_needs_normalization,
        ensure_ffmpeg,
        now_stamp,
        probe_audio_info,
        safe_mkdir,
    )

MIN_PYTHON = (3, 8)
VALID_COMPUTE_TYPES = {"int8", "int16", "float16", "float32"}

if sys.version_info < MIN_PYTHON:
    raise SystemExit("whisper-local-transcriber requiere Python 3.8 o superior.")

try:
    from argparse import BooleanOptionalAction
except ImportError:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings: List[str], dest: str, default: Optional[bool] = None, **kwargs: Any) -> None:
            if not option_strings:
                raise ValueError("BooleanOptionalAction requiere al menos un flag")
            opts = []
            for option_string in option_strings:
                opts.append(option_string)
                if option_string.startswith("--"):
                    opts.append("--no-" + option_string[2:])
            super().__init__(option_strings=opts, dest=dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: Optional[str] = None) -> None:
            setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

# -------------------- UI simple --------------------
def ui_print(msg: str = "") -> None:
    print(msg)


def ui_header() -> None:
    ui_print("\n=== whisper-local-transcriber · Release 3 ===\n")


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


def log_chunk_progress(log: Logger, processed: int, total: int, ok_count: int, failed_count: int, started_at: float) -> None:
    if total <= 0 or processed <= 0:
        return
    elapsed = time.time() - started_at
    avg = elapsed / processed
    remaining = max(0, total - processed) * avg
    pct = min(100.0, (processed / total) * 100.0)
    log.write(
        f"📊 Progreso: {processed}/{total} chunks ({pct:.0f}%) | "
        f"ok={ok_count} fallidos={failed_count} | "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
    )


# -------------------- Dependencia principal --------------------
def load_whisper_model(model_id: str, device: str, compute_type: str, cpu_threads: int, num_workers: int) -> Any:
    try:
        whisper_module = importlib.import_module("faster_whisper")
        whisper_model = getattr(whisper_module, "WhisperModel")
    except Exception as exc:
        ui_print("❌ ERROR: No se pudo importar faster_whisper. Instala deps")
        raise SystemExit(1) from exc
    return whisper_model(
        model_id,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )


# -------------------- Defaults / Aliases --------------------
DEFAULT_WORKDIR = "work"
DEFAULT_OUTDIR = "salida"
DEFAULT_LOGDIR = "logs"

# Modelo recomendado para mejor calidad en CPU (CT2 INT8 turbo).
TURBO_INT8_CT2 = "Zoont/faster-whisper-large-v3-turbo-int8-ct2"

MODEL_ALIASES = {
    # Alias cortos
    "turbo-int8": TURBO_INT8_CT2,
    "large-v3-turbo-int8": TURBO_INT8_CT2,
    "large-v3-turbo-int8-ct2": TURBO_INT8_CT2,
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "medium": "medium",
    # Si alguien pone "turbo", le damos una salida razonable (INT8 CT2)
    "turbo": TURBO_INT8_CT2,
    "large-v3-turbo": TURBO_INT8_CT2,
}

DEFAULT_MODEL = TURBO_INT8_CT2
DEFAULT_FALLBACK_MODEL = "medium"          # fallback más liviano que large turbo
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_FALLBACK_COMPUTE_TYPE = "int8"
DEFAULT_CPU_THREADS = 0
DEFAULT_NUM_WORKERS = 1
DEFAULT_LANGUAGE = "es"

DEFAULT_CHUNK_S = 45
DEFAULT_OVERLAP_S = 0.4
DEFAULT_BEAM = 1
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_NORMALIZE = True
DEFAULT_VAD_FILTER = True


# Deduplicación (fronteras de chunk)
DEDUP_WINDOW = 8
SIMILARITY_THRESHOLD = 0.92
MAX_REPEAT_PHRASE = 3

# Diarización ligera
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_TURN_GAP_S = 1.2
DEFAULT_FORCE_TURN_MAX_S = 30.0
DEFAULT_REVIEW_DIARIZATION = False


# -------------------- Texto: dedupe --------------------
def _norm_sim(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sáéíóúñü]", "", s)
    return s


def similarity_jaccard(a: str, b: str) -> float:
    a = _norm_sim(a)
    b = _norm_sim(b)
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
    if not segments:
        return []
    out: List[Dict[str, Any]] = []
    last_texts: List[str] = []
    repeat_counter: Dict[str, int] = {}

    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue

        key = _norm_sim(txt)
        is_similar = any(similarity_jaccard(txt, prev) >= SIMILARITY_THRESHOLD for prev in last_texts[-DEDUP_WINDOW:])

        if is_similar:
            repeat_counter[key] = repeat_counter.get(key, 0) + 1
            if repeat_counter[key] <= MAX_REPEAT_PHRASE:
                out.append(s)
                last_texts.append(txt)
        else:
            repeat_counter[key] = 1
            out.append(s)
            last_texts.append(txt)

    return out


# -------------------- Transcripción --------------------
def canonical_model_id(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return DEFAULT_MODEL
    low = m.lower()
    return MODEL_ALIASES.get(low, m)


def transcribe_one_chunk(
    model: Any,
    chunk_path: str,
    language: Optional[str],
    beam_size: int,
    word_timestamps: bool,
    vad_filter: bool,
) -> List[Dict[str, Any]]:
    # language: si "" o "auto" -> None
    lang = (language or "").strip().lower()
    if lang in ("", "auto", "none"):
        lang = None

    segments_iter, _info = model.transcribe(
        chunk_path,
        language=lang,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )

    out: List[Dict[str, Any]] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            out.append({"start_local": float(seg.start), "end_local": float(seg.end), "text": text})
    return out


# -------------------- Salidas + limpieza --------------------
def write_outputs(segments_global: List[Dict[str, Any]], meta: Dict[str, Any], out_txt: str, out_json: str) -> None:
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


def clean_workspace(workdir: str, logdir: str, audio_name: str) -> None:
    ui_print("\n🧹 Limpieza final (--clean): eliminando intermedios...")

    # Intermedios conocidos
    targets = [
        Path(workdir) / f"{audio_name}_estado.json",
        Path(workdir) / f"{audio_name}_partials.jsonl",
        Path(workdir) / f"{audio_name}_normalized_16k.wav",
        Path(workdir) / f"{audio_name}_chunks_metadata.json",
    ]
    for p in targets:
        if p.exists():
            try:
                p.unlink()
                ui_print(f"  ✔ Eliminado {p}")
            except Exception as e:
                ui_print(f"  ⚠️ No se pudo eliminar {p}: {e}")

    chunks_dir = Path(workdir) / f"{audio_name}_chunks"
    if chunks_dir.exists() and chunks_dir.is_dir():
        try:
            shutil.rmtree(chunks_dir)
            ui_print(f"  ✔ Eliminado {chunks_dir}")
        except Exception as e:
            ui_print(f"  ⚠️ No se pudo eliminar {chunks_dir}: {e}")

    for p in Path(logdir).glob(f"{audio_name}_*.log"):
        try:
            p.unlink()
            ui_print(f"  ✔ Eliminado {p}")
        except Exception as e:
            ui_print(f"  ⚠️ No se pudo eliminar {p}: {e}")

    ui_print("🧼 Limpieza completada.\n")


# -------------------- Asistente (opcional) --------------------
def prompt(msg: str, default: Optional[str] = None) -> str:
    q = f"{msg} [{default}]: " if default is not None else f"{msg}: "
    s = input(q).strip()
    return s if s else (default if default is not None else "")


def normalize_compute_type(value: str) -> str:
    cleaned = (value or "").strip().lower()
    if cleaned not in VALID_COMPUTE_TYPES:
        raise ValueError("compute_type inválido. Usa: int8, int16, float16 o float32.")
    return cleaned


def normalize_language(value: str) -> str:
    cleaned = (value or "").strip().lower()
    if not cleaned:
        raise ValueError("language no puede estar vacío.")
    if cleaned == "auto":
        return cleaned
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,3})?", cleaned):
        raise ValueError("language inválido. Usa 'auto' o un código como 'es', 'en' o 'pt-br'.")
    return cleaned


def normalize_non_empty_text(value: str, field_name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} no puede estar vacío.")
    return cleaned


def prompt_bool(msg: str, default: bool = True) -> bool:
    default_text = "y" if default else "n"
    valid_true = {"y", "yes", "s", "si", "sí"}
    valid_false = {"n", "no"}
    while True:
        raw = prompt(msg, default_text).strip().lower()
        if raw in valid_true:
            return True
        if raw in valid_false:
            return False
        ui_print("  ↳ Responde y/n.")


def prompt_int(msg: str, default: int, min_v: int = 1, max_v: int = 10000) -> int:
    while True:
        try:
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
            v = float(prompt(msg, str(default)))
            if v < min_v or v > max_v:
                ui_print(f"  ↳ Ingresa un valor entre {min_v} y {max_v}.")
                continue
            return v
        except Exception:
            ui_print("  ↳ Ingresa un número válido.")


def assisted_args() -> argparse.Namespace:
    ui_header()
    ui_print("\nModo asistido: completa los datos para iniciar.\n")

    while True:
        a = prompt("Ruta del audio (arrastrar/pegar)", "")
        if not a:
            ui_print("  ↳ Necesito una ruta.")
            continue
        try:
            a = resolve_audio_path(a)
            break
        except Exception as e:
            ui_print(f"  ↳ {e}")

    model = canonical_model_id(normalize_non_empty_text(prompt("Modelo (ej: turbo-int8 / medium / HF id)", DEFAULT_MODEL), "model"))

    while True:
        try:
            compute = normalize_compute_type(prompt("compute_type (int8/int16/float16/float32)", DEFAULT_COMPUTE_TYPE))
            break
        except ValueError as e:
            ui_print(f"  ↳ {e}")

    fallback = prompt("Fallback model (vacío para none)", DEFAULT_FALLBACK_MODEL).strip()
    fallback = canonical_model_id(fallback) if fallback else ""

    while True:
        try:
            fb_compute = normalize_compute_type(prompt("compute_type fallback", DEFAULT_FALLBACK_COMPUTE_TYPE))
            break
        except ValueError as e:
            ui_print(f"  ↳ {e}")

    cpu_threads = prompt_int("CPU threads (0=auto)", DEFAULT_CPU_THREADS, 0, 256)
    num_workers = prompt_int("Workers del modelo", DEFAULT_NUM_WORKERS, 1, 16)

    chunk_s = prompt_int("Chunk (seg)", DEFAULT_CHUNK_S, 10, 600)
    overlap_s = prompt_float("Overlap (seg)", DEFAULT_OVERLAP_S, 0.0, float(chunk_s) - 0.1)
    beam = prompt_int("Beam (1 recomendado)", DEFAULT_BEAM, 1, 5)

    normalize = prompt_bool("Normalizar a WAV 16k mono (y/n)", True)
    resume = prompt_bool("Reanudar si existe estado (y/n)", True)
    vad_filter = prompt_bool("VAD filter (recomendado) (y/n)", True)

    while True:
        try:
            language = normalize_language(prompt("Idioma ('es' / 'en' / 'auto')", DEFAULT_LANGUAGE))
            break
        except ValueError as e:
            ui_print(f"  ↳ {e}")

    diarize = prompt_bool("Diarización ligera (y/n)", True)
    num_speakers = DEFAULT_NUM_SPEAKERS
    turn_gap_s = DEFAULT_TURN_GAP_S
    force_turn_max_s = DEFAULT_FORCE_TURN_MAX_S
    review_diar = False
    if diarize:
        num_speakers = prompt_int("Número de participantes", DEFAULT_NUM_SPEAKERS, 1, 9)
        turn_gap_s = prompt_float("Cambio de turno si pausa ≥ (s)", DEFAULT_TURN_GAP_S, 0.1, 10.0)
        force_turn_max_s = prompt_float("Forzar cambio si bloque ≥ (s)", DEFAULT_FORCE_TURN_MAX_S, 5.0, 600.0)
        review_diar = prompt_bool("Revisar diarización al final (y/n)", False)

    workdir = DEFAULT_WORKDIR
    outdir = DEFAULT_OUTDIR
    logdir = DEFAULT_LOGDIR

    force_reuse_chunks = prompt_bool("Forzar reutilización chunks (y/n)", False)
    clean = prompt_bool("Eliminar intermedios al final (--clean) (y/n)", False)

    args = argparse.Namespace(
        audio=a,
        workdir=workdir, outdir=outdir, logdir=logdir,
        model=model, compute_type=compute,
        fallback_model=fallback, fallback_compute_type=fb_compute,
        cpu_threads=cpu_threads, num_workers=num_workers,
        language=language,
        device=DEFAULT_DEVICE,
        chunk_s=chunk_s, overlap_s=overlap_s,
        beam=beam, word_timestamps=False,
        normalize=normalize, resume=resume,
        vad_filter=vad_filter,
        diarize=diarize, num_speakers=num_speakers, turn_gap_s=turn_gap_s,
        force_turn_max_s=force_turn_max_s, review_diarization=review_diar,
        force_reuse_chunks=force_reuse_chunks,
        clean=clean,
    )

    ui_print("\nResumen:")
    ui_print(f"- Audio: {args.audio}")
    ui_print(f"- Modelo: {args.model} (compute: {args.compute_type})")
    ui_print(f"- CPU threads: {args.cpu_threads} | workers: {args.num_workers}")
    ui_print(f"- Fallback: {args.fallback_model or '(none)'}")
    ui_print(f"- Idioma: {args.language} | VAD: {'Sí' if args.vad_filter else 'No'}")
    ui_print(f"- Chunk: {args.chunk_s}s | overlap: {args.overlap_s}s | beam: {args.beam}")
    ui_print(f"- Diarize: {'Sí' if args.diarize else 'No'}")

    ok_go = prompt_bool("¿Iniciar? (y/n)", True)
    if not ok_go:
        ui_print("Ejecución cancelada.")
        raise SystemExit(0)

    return args


# -------------------- Validación args --------------------
def validate_args(args: argparse.Namespace) -> None:
    args.model = canonical_model_id(normalize_non_empty_text(args.model, "model"))
    args.compute_type = normalize_compute_type(args.compute_type)
    args.language = normalize_language(args.language)
    args.workdir = normalize_non_empty_text(args.workdir, "workdir")
    args.outdir = normalize_non_empty_text(args.outdir, "outdir")
    args.logdir = normalize_non_empty_text(args.logdir, "logdir")
    args.audio = normalize_non_empty_text(args.audio, "audio")
    args.fallback_model = canonical_model_id((args.fallback_model or "").strip()) if args.fallback_model else ""
    args.fallback_compute_type = normalize_compute_type(args.fallback_compute_type)

    if args.chunk_s <= 0:
        raise ValueError("chunk_s debe ser > 0")
    if args.overlap_s < 0:
        raise ValueError("overlap_s debe ser >= 0")
    if args.overlap_s >= args.chunk_s:
        raise ValueError("overlap_s debe ser menor que chunk_s")
    if args.beam < 1 or args.beam > 5:
        raise ValueError("beam debe estar entre 1 y 5")
    if args.cpu_threads < 0:
        raise ValueError("cpu_threads debe ser >= 0")
    if args.num_workers < 1:
        raise ValueError("num_workers debe ser >= 1")
    if args.turn_gap_s <= 0:
        raise ValueError("turn_gap_s debe ser > 0")
    if args.force_turn_max_s <= 0:
        raise ValueError("force_turn_max_s debe ser > 0")
    if args.diarize and (args.num_speakers < 1 or args.num_speakers > 9):
        raise ValueError("num_speakers debe estar entre 1 y 9")


# -------------------- Pipeline principal --------------------
def run_pipeline(args: argparse.Namespace) -> None:
    validate_args(args)
    ensure_ffmpeg(ui_print)

    safe_mkdir(args.workdir)
    safe_mkdir(args.outdir)
    safe_mkdir(args.logdir)

    audio_path = resolve_audio_path(args.audio)
    audio_name = Path(audio_path).stem

    log_path = str(Path(args.logdir) / f"{audio_name}_{now_stamp()}.log")
    log = Logger(log_path, ui_print)

    # Paths
    state_path = str(Path(args.workdir) / f"{audio_name}_estado.json")
    chunks_dir = str(Path(args.workdir) / f"{audio_name}_chunks")
    normalized_wav = str(Path(args.workdir) / f"{audio_name}_normalized_16k.wav")
    partials_jsonl = str(Path(args.workdir) / f"{audio_name}_partials.jsonl")
    chunks_metadata_file = str(Path(args.workdir) / f"{audio_name}_chunks_metadata.json")

    out_txt = str(Path(args.outdir) / f"{audio_name}_transcripcion_final.txt")
    out_json = str(Path(args.outdir) / f"{audio_name}_transcripcion_final.json")

    # Canonical model ids
    args.model = canonical_model_id(args.model)
    args.fallback_model = canonical_model_id(args.fallback_model) if args.fallback_model else ""

    # Log config
    log.write(f"📌 Audio: {audio_path}")
    log.write(f"🧠 Modelo: {args.model} | compute={args.compute_type} | device={args.device}")
    log.write(f"🧵 CPU threads={args.cpu_threads} | workers={args.num_workers}")
    log.write(f"🧠 Fallback: {args.fallback_model or '(none)'} | compute={args.fallback_compute_type or '(none)'}")
    log.write(f"⚙️ chunk={args.chunk_s}s overlap={args.overlap_s}s beam={args.beam} word_ts={bool(args.word_timestamps)} normalize={bool(args.normalize)} resume={bool(args.resume)} vad={bool(args.vad_filter)}")
    log.write(f"🗣️ diarize={bool(args.diarize)} speakers={getattr(args,'num_speakers',0)} turn_gap={getattr(args,'turn_gap_s',0)} force_turn_max={getattr(args,'force_turn_max_s',0)} review={getattr(args,'review_diarization',False)}")
    log.write(f"🧾 Log file: {log_path}")

    # Resume state
    state = load_state(state_path) if args.resume else {}

    # Firma de config para resume (evita mezclar cosas)
    cfg_for_sig = {
        "audio_path": audio_path,
        "model": args.model,
        "fallback_model": args.fallback_model,
        "compute_type": args.compute_type,
        "fallback_compute_type": args.fallback_compute_type if args.fallback_model else "",
        "cpu_threads": args.cpu_threads,
        "num_workers": args.num_workers,
        "language": args.language,
        "chunk_s": args.chunk_s,
        "overlap_s": args.overlap_s,
        "beam": args.beam,
        "word_timestamps": bool(args.word_timestamps),
        "normalize": bool(args.normalize),
        "vad_filter": bool(args.vad_filter),
    }
    sig = config_signature(cfg_for_sig)

    if state and args.resume:
        if state.get("config_sig") != sig:
            if getattr(args, "force_reuse_chunks", False):
                log.write("⚠️ config_sig difiere, pero --force_reuse_chunks está activo. Continuaré con cautela.")
            else:
                log.write("⚠️ Estado incompatible con la configuración actual. Reiniciando estado (se ignora --resume).")
                state = {}
                # Limpiar parciales para evitar mezcla
                try:
                    if os.path.exists(partials_jsonl):
                        os.remove(partials_jsonl)
                        log.write(f"🧹 Eliminado parcial: {partials_jsonl}")
                except Exception as e:
                    log.write(f"⚠️ No pude borrar parcial: {e}")

    if not state:
        state = {
            "audio_path": audio_path,
            "audio_name": audio_name,
            "config_sig": sig,
            "completed_chunks": [],
            "failed_chunks": [],
            "total_chunks": None,
        }
        save_state(state_path, state)
        log.write(f"🧾 Estado creado: {state_path}")

    # Normalize (opcional)
    audio_info = probe_audio_info(audio_path)
    input_duration_s = float(audio_info.get("duration_s") or 0.0)
    log.write(
        f"🎧 Audio detectado: duration={format_duration(input_duration_s)} "
        f"sample_rate={int(audio_info.get('sample_rate') or 0)}Hz "
        f"channels={int(audio_info.get('channels') or 0)}"
    )
    input_for_split = audio_path
    if args.normalize:
        if os.path.exists(normalized_wav):
            log.write("♻️ Reutilizando WAV normalizado existente.")
            input_for_split = normalized_wav
        elif audio_needs_normalization(audio_info):
            log.write("🔧 Normalizando a WAV 16k mono...")
            normalize_to_wav_16k_mono(audio_path, normalized_wav)
            input_for_split = normalized_wav
        else:
            log.write("✅ El audio ya está en 16 kHz mono; se reutiliza sin re-normalizar.")
            input_for_split = audio_path

    # Chunks: reuso si metadata coincide
    safe_mkdir(chunks_dir)
    chunks: List[Chunk] = []

    can_reuse = False
    existing = sorted(Path(chunks_dir).glob(f"{audio_name}_chunk_*.wav"))
    meta_chunks_obj: Dict[str, Any] = {}

    if existing and os.path.exists(chunks_metadata_file):
        try:
            with open(chunks_metadata_file, "r", encoding="utf-8") as f:
                meta_chunks_obj = json.load(f)
            if (
                meta_chunks_obj.get("chunk_s") == args.chunk_s
                and meta_chunks_obj.get("overlap_s") == args.overlap_s
                and meta_chunks_obj.get("input_path") == input_for_split
            ):
                can_reuse = True
        except Exception:
            can_reuse = False

    if not can_reuse and getattr(args, "force_reuse_chunks", False) and existing and os.path.exists(chunks_metadata_file):
        log.write("⚠️ Forzando reutilización de chunks (--force_reuse_chunks).")
        try:
            with open(chunks_metadata_file, "r", encoding="utf-8") as f:
                meta_chunks_obj = json.load(f)
            can_reuse = True
        except Exception:
            can_reuse = False

    if can_reuse:
        try:
            for item in meta_chunks_obj.get("chunks", []):
                chunks.append(Chunk(
                    idx=int(item["idx"]),
                    path=str(item["path"]),
                    start_s=float(item["start_s"]),
                    end_s=float(item["end_s"]),
                ))
            chunks.sort(key=lambda c: c.idx)
            log.write(f"♻️ Reutilizando {len(chunks)} chunks existentes.")
        except Exception as e:
            log.write(f"⚠️ Metadata corrupta ({e}). Regenerando chunks...")
            chunks = []

    if not chunks:
        log.write("✂️ Generando chunks...")
        chunks = split_audio_fixed(input_for_split, chunks_dir, int(args.chunk_s), float(args.overlap_s), audio_name, input_duration_s)
        log.write(f"✅ Chunks generados: {len(chunks)}")

        meta_chunks_obj = {
            "chunk_s": args.chunk_s,
            "overlap_s": args.overlap_s,
            "input_path": input_for_split,
            "duration_s": input_duration_s,
            "chunks": [{"idx": ch.idx, "path": ch.path, "start_s": ch.start_s, "end_s": ch.end_s} for ch in chunks],
        }
        with open(chunks_metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_chunks_obj, f, ensure_ascii=False, indent=2)
        log.write(f"💾 Metadata de chunks: {chunks_metadata_file}")

    state["total_chunks"] = len(chunks)
    save_state(state_path, state)

    # Cargar modelos
    log.write("🧠 Cargando modelo principal...")
    model_main = load_whisper_model(args.model, args.device, args.compute_type, args.cpu_threads, args.num_workers)

    model_fallback = None
    fallback_available = bool(args.fallback_model and args.fallback_model != args.model)

    completed = set(state.get("completed_chunks", []))
    failed = set(state.get("failed_chunks", []))
    pending_count = len([ch for ch in chunks if ch.idx not in completed])
    log.write(f"📦 Chunks: total={len(chunks)} completados={len(completed)} pendientes={pending_count} fallidos_previos={len(failed)}")

    if not os.path.exists(partials_jsonl):
        open(partials_jsonl, "w", encoding="utf-8").close()

    t0 = time.time()

    for ch in chunks:
        if ch.idx in completed:
            log.write(f"↪️ Saltando chunk {ch.idx} (ya completado)")
            continue

        log.write(f"\n▶️ Chunk {ch.idx+1}/{len(chunks)}  [{ch.start_s/60:.1f}m -> {ch.end_s/60:.1f}m]")

        def persist_failure(err: str) -> None:
            failed.add(ch.idx)
            state["failed_chunks"] = sorted(failed)
            save_state(state_path, state)
            log.write(f"⚠️ Fallido: chunk {ch.idx} | {err}")
            log_chunk_progress(log, len(completed.union(failed)), len(chunks), len(completed), len(failed), t0)

        try:
            local_segments = transcribe_one_chunk(
                model_main, ch.path,
                args.language, int(args.beam),
                bool(args.word_timestamps),
                bool(args.vad_filter),
            )
        except Exception as e:
            log.write(f"⚠️ Falló con principal: {e}")
            if not fallback_available:
                persist_failure(str(e))
                continue
            if model_fallback is None:
                log.write("🧠 Cargando fallback...")
                model_fallback = load_whisper_model(
                    args.fallback_model,
                    args.device,
                    args.fallback_compute_type,
                    args.cpu_threads,
                    args.num_workers,
                )
            log.write("🔁 Reintentando con fallback...")
            try:
                local_segments = transcribe_one_chunk(
                    model_fallback, ch.path,
                    args.language, int(args.beam),
                    bool(args.word_timestamps),
                    bool(args.vad_filter),
                )
            except Exception as e2:
                persist_failure(str(e2))
                continue

        segs_global: List[Dict[str, Any]] = [{
            "chunk_idx": ch.idx,
            "start": float(ch.start_s + s["start_local"]),
            "end": float(ch.start_s + s["end_local"]),
            "text": s["text"],
        } for s in local_segments]

        with open(partials_jsonl, "a", encoding="utf-8") as f:
            for sg in segs_global:
                f.write(json.dumps(sg, ensure_ascii=False) + "\n")

        completed.add(ch.idx)
        state["completed_chunks"] = sorted(completed)
        save_state(state_path, state)

        log.write(f"✅ Chunk {ch.idx} OK ({len(segs_global)} segs)")
        log_chunk_progress(log, len(completed.union(failed)), len(chunks), len(completed), len(failed), t0)

    # Consolidar parciales
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

    # Diarización
    segments_for_diar = segments_global
    if args.diarize:
        diar = diarize_light(segments_for_diar, int(args.num_speakers), float(args.turn_gap_s), float(args.force_turn_max_s))
        if getattr(args, "review_diarization", False):
            diar = review_diarization_interactive(diar, int(args.num_speakers), ui_print)
        segments_final = diar
    else:
        segments_final = segments_for_diar

    # Meta
    t1 = time.time()
    meta = {
        "audio": audio_name,
        "audio_path": audio_path,
        "model": args.model,
        "fallback_model": args.fallback_model,
        "compute_type": args.compute_type,
        "fallback_compute_type": args.fallback_compute_type if args.fallback_model else "",
        "cpu_threads": args.cpu_threads,
        "num_workers": args.num_workers,
        "language": args.language,
        "device": args.device,
        "chunk_s": args.chunk_s,
        "overlap_s": args.overlap_s,
        "beam": args.beam,
        "word_timestamps": bool(args.word_timestamps),
        "audio_duration_s": input_duration_s,
        "normalized": bool(args.normalize),
        "vad_filter": bool(args.vad_filter),
        "diarize": bool(args.diarize),
        "num_speakers": int(args.num_speakers),
        "turn_gap_s": float(args.turn_gap_s),
        "force_turn_max_s": float(args.force_turn_max_s),
        "review_diarization": bool(getattr(args, "review_diarization", False)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_total": len(chunks),
        "chunks_ok": len(completed),
        "chunks_failed": len(failed),
        "log": log_path,
        "elapsed_min": round((t1 - t0) / 60.0, 2),
    }

    write_outputs(segments_final, meta, out_txt, out_json)
    log.write("\n✅ Listo.")
    log.write(f"📊 Resumen final: chunks_ok={len(completed)}/{len(chunks)} chunks_failed={len(failed)} elapsed={format_duration(t1 - t0)}")
    log.write(f"TXT:  {out_txt}")
    log.write(f"JSON: {out_json}")
    if failed:
        log.write(f"⚠️ Chunks fallidos: {sorted(failed)}")
    log.write(f"Tiempo total: {(t1 - t0)/60:.1f} min\n")

    if getattr(args, "clean", False):
        clean_workspace(args.workdir, args.logdir, audio_name)


# -------------------- CLI --------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="whisper-local-transcriber (Release 3). Si omites audio, entra en asistente.")
    p.add_argument("audio", nargs="?", default=None, help="Ruta del audio. Si omites, entra en asistente.")

    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--logdir", default=DEFAULT_LOGDIR)

    # Modelos: aceptar string libre (HF id / ruta / alias)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Modelo: alias (turbo-int8) o HuggingFace id/ruta.")
    p.add_argument("--fallback_model", default=DEFAULT_FALLBACK_MODEL, help="Fallback (vacío para none).")

    p.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu"], help="Este release prioriza CPU.")
    p.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE, choices=["int8", "int16", "float16", "float32"])
    p.add_argument("--fallback_compute_type", default=DEFAULT_FALLBACK_COMPUTE_TYPE, choices=["int8", "int16", "float16", "float32"])
    p.add_argument("--cpu_threads", type=int, default=DEFAULT_CPU_THREADS, help="Threads CPU para faster-whisper (0=auto).")
    p.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Workers internos del modelo.")

    p.add_argument("--language", default=DEFAULT_LANGUAGE, help="Idioma: es/en/auto")
    p.add_argument("--chunk_s", type=int, default=DEFAULT_CHUNK_S)
    p.add_argument("--overlap_s", type=float, default=DEFAULT_OVERLAP_S)
    p.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    p.add_argument("--word_timestamps", action=BooleanOptionalAction, default=DEFAULT_WORD_TIMESTAMPS)
    p.add_argument("--normalize", action=BooleanOptionalAction, default=DEFAULT_NORMALIZE)
    p.add_argument("--resume", action=BooleanOptionalAction, default=True)
    p.add_argument("--vad_filter", action=BooleanOptionalAction, default=DEFAULT_VAD_FILTER)

    p.add_argument("--diarize", action=BooleanOptionalAction, default=DEFAULT_DIARIZE)
    p.add_argument("--num_speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    p.add_argument("--turn_gap_s", type=float, default=DEFAULT_TURN_GAP_S)
    p.add_argument("--force_turn_max_s", type=float, default=DEFAULT_FORCE_TURN_MAX_S)
    p.add_argument("--review_diarization", action=BooleanOptionalAction, default=DEFAULT_REVIEW_DIARIZATION)

    p.add_argument("--force_reuse_chunks", action=BooleanOptionalAction, default=False, help="Reusar chunks aunque config difiera (cautela).")
    p.add_argument("--clean", action=BooleanOptionalAction, default=False, help="Eliminar intermedios al final.")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.audio is None:
        args = assisted_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()