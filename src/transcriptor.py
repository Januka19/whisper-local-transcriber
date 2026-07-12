#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
whisper-local-transcriber :: Transcriptor Release 3 (estable)

Objetivo:
- Offline, CPU-friendly, audios largos, ejecución reproducible.
- Resume real (estado + parciales), con validación de compatibilidad.
- Diarización local mediante modelo Sherpa ONNX.
- Modelos flexibles: acepta IDs de HuggingFace / rutas locales / alias.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.audio_utils import Chunk, normalize_to_wav_16k_mono, split_audio_fixed
    from src.diarization_utils import (
        DEFAULT_DIARIZATION_MODEL_DIR,
        create_sherpa_diarizer,
        diarize_sherpa,
        require_sherpa_diarization,
        review_diarization_interactive,
        setup_sherpa_diarization_models,
    )
    from src.state_utils import audio_source_identity, config_signature, load_state, resolve_audio_path, save_state
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
    from diarization_utils import (
        DEFAULT_DIARIZATION_MODEL_DIR,
        create_sherpa_diarizer,
        diarize_sherpa,
        require_sherpa_diarization,
        review_diarization_interactive,
        setup_sherpa_diarization_models,
    )
    from state_utils import audio_source_identity, config_signature, load_state, resolve_audio_path, save_state
    from system_utils import (
        Logger,
        audio_needs_normalization,
        ensure_ffmpeg,
        now_stamp,
        probe_audio_info,
        safe_mkdir,
    )

MIN_PYTHON = (3, 9)
VALID_COMPUTE_TYPES = {"int8", "int16", "float16", "float32"}

if sys.version_info < MIN_PYTHON:
    raise SystemExit("whisper-local-transcriber requiere Python 3.9 o superior.")

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

# Alias nativo soportado por faster-whisper. La librería resuelve este nombre a
# su conversión CTranslate2 mantenida; compute_type=int8 conserva el perfil CPU.
DEFAULT_TURBO_MODEL = "large-v3-turbo"
LEGACY_TURBO_INT8_CT2 = "Zoont/faster-whisper-large-v3-turbo-int8-ct2"

MODEL_ALIASES = {
    # Alias cortos
    "turbo-int8": DEFAULT_TURBO_MODEL,
    "large-v3-turbo-int8": DEFAULT_TURBO_MODEL,
    "large-v3-turbo-int8-ct2": DEFAULT_TURBO_MODEL,
    "large-v3": "large-v3",
    "quality": "large-v3",
    "max-quality": "large-v3",
    "large-v2": "large-v2",
    "medium": "medium",
    "turbo": DEFAULT_TURBO_MODEL,
    "large-v3-turbo": DEFAULT_TURBO_MODEL,
    # Permite seguir usando el checkpoint anterior ya descargado en entornos
    # sin red, pero deja de ser la recomendación por defecto.
    "turbo-int8-legacy": LEGACY_TURBO_INT8_CT2,
}

DEFAULT_MODEL = DEFAULT_TURBO_MODEL
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

# Diarización por modelo
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = 2
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


def sanitize_partials(partials_jsonl: str, completed_chunks: set[int]) -> tuple[int, set[int]]:
    """Remove unsafe records and identify completed chunks that need a retry."""
    path = Path(partials_jsonl)
    if not path.exists():
        return 0, set()

    valid_records: List[tuple[int, Dict[str, Any]]] = []
    removed = 0
    retry_chunks: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            chunk_idx: Optional[int] = None
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("el parcial no es un objeto")
                chunk_idx = int(record["chunk_idx"])
                start = float(record["start"])
                end = float(record["end"])
                if not math.isfinite(start) or not math.isfinite(end) or end < start:
                    raise ValueError("marcas de tiempo inválidas")
                if not isinstance(record["text"], str):
                    raise TypeError("texto inválido")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                removed += 1
                if chunk_idx is None:
                    # Sin índice no se puede saber qué resultado quedó dañado.
                    retry_chunks.update(completed_chunks)
                elif chunk_idx in completed_chunks:
                    retry_chunks.add(chunk_idx)
                continue
            if chunk_idx not in completed_chunks:
                removed += 1
                continue
            valid_records.append((chunk_idx, record))

    kept = [
        json.dumps(record, ensure_ascii=False)
        for chunk_idx, record in valid_records
        if chunk_idx not in retry_chunks
    ]
    removed += len(valid_records) - len(kept)
    if removed or retry_chunks:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for line in kept:
                output.write(line + "\n")
        temporary.replace(path)
    return removed, retry_chunks


def clean_workspace(workdir: str, logdir: str, audio_name: str) -> None:
    ui_print("\n🧹 Limpieza final (--clean): eliminando intermedios...")

    # Intermedios conocidos
    targets = [
        Path(workdir) / f"{audio_name}_estado.json",
        Path(workdir) / f"{audio_name}_partials.jsonl",
        Path(workdir) / f"{audio_name}_normalized_16k.wav",
        Path(workdir) / f"{audio_name}_normalized_16k.json",
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

    diarize = prompt_bool("Diarización con modelo Sherpa (y/n)", True)
    num_speakers = DEFAULT_NUM_SPEAKERS
    review_diar = False
    if diarize:
        num_speakers = prompt_int("Número de participantes", DEFAULT_NUM_SPEAKERS, 1, 9)
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
        diarize=diarize, num_speakers=num_speakers,
        review_diarization=review_diar,
        diarization_model_dir=DEFAULT_DIARIZATION_MODEL_DIR,
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
    args.diarization_model_dir = normalize_non_empty_text(
        getattr(args, "diarization_model_dir", DEFAULT_DIARIZATION_MODEL_DIR),
        "diarization_model_dir",
    )

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
    normalized_metadata_file = str(Path(args.workdir) / f"{audio_name}_normalized_16k.json")

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
    log.write(f"🗣️ diarize={bool(args.diarize)} speakers={getattr(args,'num_speakers',0)} review={getattr(args,'review_diarization',False)} model_dir={args.diarization_model_dir}")
    log.write(f"🧾 Log file: {log_path}")

    diarization_threads = int(args.cpu_threads) if int(args.cpu_threads) > 0 else min(4, os.cpu_count() or 1)
    sherpa_diarizer = None
    if args.diarize:
        readiness = require_sherpa_diarization(args.diarization_model_dir)
        sherpa_diarizer = create_sherpa_diarizer(
            args.diarization_model_dir,
            int(args.num_speakers),
            diarization_threads,
        )
        log.write(f"✅ Diarización Sherpa preparada: {readiness}")

    source_identity = audio_source_identity(audio_path)

    # Resume state
    state = load_state(state_path) if args.resume else {}

    # Firma de config para resume (evita mezclar cosas)
    cfg_for_sig = {
        "audio_path": audio_path,
        "audio_source": source_identity,
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
            log.write("⚠️ Estado incompatible con la configuración actual. Se reiniciarán estado y parciales.")
            state = {}
            # --force_reuse_chunks solo aplica al audio troceado, nunca a
            # resultados de transcripción creados con otra configuración.
            try:
                if os.path.exists(partials_jsonl):
                    os.remove(partials_jsonl)
                    log.write(f"🧹 Eliminado parcial: {partials_jsonl}")
            except OSError as e:
                raise RuntimeError(f"No se pudo eliminar el parcial incompatible: {e}") from e

    if not state:
        # Sin un estado válido no hay forma segura de asociar parciales previos
        # con esta ejecución (también cubre --no-resume y estados corruptos).
        if os.path.exists(partials_jsonl):
            try:
                os.remove(partials_jsonl)
                log.write(f"🧹 Inicio limpio: eliminado parcial anterior {partials_jsonl}")
            except OSError as e:
                raise RuntimeError(f"No se pudo reiniciar el archivo de parciales: {e}") from e
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
    elif state.get("completed_chunks") and not os.path.exists(partials_jsonl):
        # Saltar chunks completados sin sus resultados produciría una salida
        # final incompleta. Es preferible retranscribirlos.
        log.write("⚠️ Faltan los parciales del estado reanudable; se retranscribirán los chunks completados.")
        state["completed_chunks"] = []
        state["failed_chunks"] = []
        save_state(state_path, state)

    # Normalize (opcional)
    audio_info = probe_audio_info(audio_path)
    input_duration_s = float(audio_info.get("duration_s") or 0.0)
    log.write(
        f"🎧 Audio detectado: duration={format_duration(input_duration_s)} "
        f"sample_rate={int(audio_info.get('sample_rate') or 0)}Hz "
        f"channels={int(audio_info.get('channels') or 0)}"
    )
    input_for_split = audio_path
    normalized_is_current = False
    if os.path.exists(normalized_wav) and os.path.exists(normalized_metadata_file):
        try:
            with open(normalized_metadata_file, "r", encoding="utf-8") as f:
                normalized_is_current = json.load(f).get("audio_source") == source_identity
        except (OSError, ValueError, TypeError):
            normalized_is_current = False

    if args.normalize:
        if normalized_is_current:
            log.write("♻️ Reutilizando WAV normalizado existente.")
            input_for_split = normalized_wav
        elif audio_needs_normalization(audio_info):
            if os.path.exists(normalized_wav):
                log.write("⚠️ El WAV normalizado pertenece a otra versión del audio; se regenerará.")
            log.write("🔧 Normalizando a WAV 16k mono...")
            normalize_to_wav_16k_mono(audio_path, normalized_wav)
            with open(normalized_metadata_file, "w", encoding="utf-8") as f:
                json.dump({"audio_source": source_identity}, f, ensure_ascii=False, indent=2)
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
                and meta_chunks_obj.get("audio_source") == source_identity
            ):
                listed_chunks = meta_chunks_obj.get("chunks") or []
                can_reuse = bool(listed_chunks) and all(Path(str(item.get("path", ""))).is_file() for item in listed_chunks)
        except Exception:
            can_reuse = False

    if not can_reuse and getattr(args, "force_reuse_chunks", False) and existing and os.path.exists(chunks_metadata_file):
        log.write("⚠️ Forzando reutilización de chunks (--force_reuse_chunks).")
        try:
            with open(chunks_metadata_file, "r", encoding="utf-8") as f:
                meta_chunks_obj = json.load(f)
            listed_chunks = meta_chunks_obj.get("chunks") or []
            can_reuse = bool(listed_chunks) and all(
                Path(str(item.get("path", ""))).is_file() for item in listed_chunks
            )
            if not can_reuse:
                log.write("⚠️ Los chunks forzados están incompletos; se regenerarán.")
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
        for stale_chunk in Path(chunks_dir).glob(f"{audio_name}_chunk_*.wav"):
            stale_chunk.unlink()
        chunks = split_audio_fixed(input_for_split, chunks_dir, int(args.chunk_s), float(args.overlap_s), audio_name, input_duration_s)
        log.write(f"✅ Chunks generados: {len(chunks)}")

        if not chunks:
            raise RuntimeError("No se pudieron generar chunks. Verifica que el archivo contenga audio válido.")

        meta_chunks_obj = {
            "chunk_s": args.chunk_s,
            "overlap_s": args.overlap_s,
            "input_path": input_for_split,
            "audio_source": source_identity,
            "duration_s": input_duration_s,
            "chunks": [{"idx": ch.idx, "path": ch.path, "start_s": ch.start_s, "end_s": ch.end_s} for ch in chunks],
        }
        with open(chunks_metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_chunks_obj, f, ensure_ascii=False, indent=2)
        log.write(f"💾 Metadata de chunks: {chunks_metadata_file}")

    state["total_chunks"] = len(chunks)
    save_state(state_path, state)

    try:
        completed = {int(idx) for idx in (state.get("completed_chunks") or [])}
        failed = {int(idx) for idx in (state.get("failed_chunks") or [])}
    except (TypeError, ValueError):
        completed = set()
        failed = set()
        state["completed_chunks"] = []
        state["failed_chunks"] = []
        save_state(state_path, state)
        log.write("⚠️ Las listas de chunks del estado eran inválidas; se reiniciarán.")
    valid_chunk_ids = {chunk.idx for chunk in chunks}
    invalid_state_ids = completed.union(failed).difference(valid_chunk_ids)
    if invalid_state_ids:
        completed.intersection_update(valid_chunk_ids)
        failed.intersection_update(valid_chunk_ids)
        state["completed_chunks"] = sorted(completed)
        state["failed_chunks"] = sorted(failed)
        save_state(state_path, state)
        log.write(f"🧹 Estado corregido: descartados chunks inexistentes {sorted(invalid_state_ids)}.")

    stale_failures = completed.intersection(failed)
    if stale_failures:
        failed.difference_update(stale_failures)
        state["failed_chunks"] = sorted(failed)
        save_state(state_path, state)
        log.write(f"🧹 Estado corregido: {len(stale_failures)} chunks completados dejaron de figurar como fallidos.")

    removed_partials, retry_chunks = sanitize_partials(partials_jsonl, completed)
    if retry_chunks:
        completed.difference_update(retry_chunks)
        failed.difference_update(retry_chunks)
        state["completed_chunks"] = sorted(completed)
        state["failed_chunks"] = sorted(failed)
        save_state(state_path, state)
        log.write(f"⚠️ Parciales corruptos: se retranscribirán los chunks {sorted(retry_chunks)}.")
    if removed_partials:
        log.write(f"🧹 Parciales corregidos: descartados {removed_partials} registros no confirmados.")

    pending_count = len([ch for ch in chunks if ch.idx not in completed])
    log.write(f"📦 Chunks: total={len(chunks)} completados={len(completed)} pendientes={pending_count} fallidos_previos={len(failed)}")

    if not os.path.exists(partials_jsonl):
        open(partials_jsonl, "w", encoding="utf-8").close()

    model_main = None
    if pending_count:
        log.write("🧠 Cargando modelo principal...")
        model_main = load_whisper_model(args.model, args.device, args.compute_type, args.cpu_threads, args.num_workers)
    else:
        log.write("♻️ Todos los chunks ya están completos; no se carga el modelo de transcripción.")

    model_fallback = None
    fallback_available = bool(args.fallback_model and args.fallback_model != args.model)

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
        failed.discard(ch.idx)
        state["completed_chunks"] = sorted(completed)
        state["failed_chunks"] = sorted(failed)
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
        if not normalized_is_current:
            log.write("🔧 Preparando WAV PCM 16 kHz mono para diarización Sherpa...")
            normalize_to_wav_16k_mono(audio_path, normalized_wav)
            with open(normalized_metadata_file, "w", encoding="utf-8") as f:
                json.dump({"audio_source": source_identity}, f, ensure_ascii=False, indent=2)
        diar = diarize_sherpa(
            normalized_wav,
            segments_for_diar,
            int(args.num_speakers),
            args.diarization_model_dir,
            diarization_threads,
            sherpa_diarizer,
        )
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
        "diarization_backend": "sherpa" if args.diarize else "disabled",
        "diarization_model_dir": args.diarization_model_dir if args.diarize else "",
        "num_speakers": int(args.num_speakers),
        "review_diarization": bool(getattr(args, "review_diarization", False)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_total": len(chunks),
        "chunks_ok": len(completed),
        "chunks_failed": len(failed),
        "log": log_path,
        "elapsed_min": round((t1 - t0) / 60.0, 2),
    }

    write_outputs(segments_final, meta, out_txt, out_json)
    if failed:
        log.write("\n⚠️ Finalizado con errores; las salidas no incluyen los chunks fallidos.")
    else:
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
    p.add_argument(
        "--diarization_model_dir",
        default=DEFAULT_DIARIZATION_MODEL_DIR,
        help="Directorio local de pesos de diarización.",
    )
    p.add_argument(
        "--setup_diarization_models",
        action="store_true",
        help="Descarga explícitamente los pesos ligeros de Sherpa y sale.",
    )
    p.add_argument("--num_speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    p.add_argument("--review_diarization", action=BooleanOptionalAction, default=DEFAULT_REVIEW_DIARIZATION)

    p.add_argument("--force_reuse_chunks", action=BooleanOptionalAction, default=False, help="Reusar chunks aunque config difiera (cautela).")
    p.add_argument("--clean", action=BooleanOptionalAction, default=False, help="Eliminar intermedios al final.")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.setup_diarization_models:
        try:
            paths = setup_sherpa_diarization_models(args.diarization_model_dir, ui_print)
            require_sherpa_diarization(args.diarization_model_dir)
            create_sherpa_diarizer(args.diarization_model_dir, DEFAULT_NUM_SPEAKERS)
            ui_print("✅ Modelos de diarización listos:")
            for name, path in paths.items():
                ui_print(f"- {name}: {path}")
        except Exception as exc:
            ui_print(f"❌ ERROR preparando modelos de diarización: {exc}")
            raise SystemExit(1)
        return
    if args.audio is None:
        args = assisted_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        ui_print("\n⚠️ Ejecución interrumpida. Puedes reanudarla con la misma configuración.")
        raise SystemExit(130)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        ui_print(f"❌ ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
