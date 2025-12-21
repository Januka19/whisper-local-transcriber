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

try:
    from faster_whisper import WhisperModel
except Exception:
    print("ERROR: No se pudo importar faster_whisper. Instala con: pip install -U faster-whisper")
    sys.exit(1)

DEFAULT_WORKDIR = "work"
DEFAULT_OUTDIR = "salida"
DEFAULT_LOGDIR = "logs"

DEFAULT_MODEL = "medium"
DEFAULT_FALLBACK_MODEL = "small"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_LANGUAGE = "es"

DEFAULT_CHUNK_S = 60
DEFAULT_OVERLAP_S = 0.5
DEFAULT_BEAM = 1
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_NORMALIZE = True

# Post-procesado (Paso 1)
DEFAULT_POSTPROCESS = True
DEFAULT_REMOVE_FILLERS = True
DEFAULT_WRITE_CLEAN = True
DEFAULT_MERGE_GAP_S = 0.8

# Dedup
DEDUP_WINDOW = 8
SIMILARITY_THRESHOLD = 0.92
MAX_REPEAT_PHRASE = 3

# Diarización ligera
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_TURN_GAP_S = 1.2
DEFAULT_FORCE_TURN_MAX_S = 30.0
DEFAULT_REVIEW_DIARIZATION = False

@dataclass
class Chunk:
    idx: int
    path: str
    start_s: float
    end_s: float

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ERROR: No se encontró ffmpeg/ffprobe en PATH. Instala con: sudo apt-get install -y ffmpeg")
        sys.exit(1)

def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Comando falló: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def ffprobe_duration_seconds(path: str) -> float:
    cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe falló para {path}\n{p.stderr}")
    return float(p.stdout.strip())

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def write_log(log_path: str, msg: str) -> None:
    safe_mkdir(os.path.dirname(log_path))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(msg, flush=True)

def normalize_to_wav_16k_mono(input_audio: str, out_wav: str) -> None:
    cmd = ["ffmpeg","-y","-i",input_audio,"-ac","1","-ar","16000","-vn","-af","highpass=f=80,volume=1.2",out_wav]
    run(cmd)

def split_audio_fixed(input_audio: str, workdir: str, chunk_s: int, overlap_s: float, prefix: str) -> List[Chunk]:
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
        out_path = os.path.join(workdir, f"{prefix}_chunk_{idx:04d}.wav")
        cmd = ["ffmpeg","-y","-ss",f"{start}","-i",input_audio,"-t",f"{end-start}","-ac","1","-ar","16000","-vn",out_path]
        run(cmd)
        chunks.append(Chunk(idx=idx, path=out_path, start_s=start, end_s=end))
        idx += 1
        t += step
    return chunks

def resolve_audio_path(audio_arg: str) -> str:
    p = Path(audio_arg).expanduser()
    if p.exists():
        return str(p)
    p2 = Path(os.getcwd()) / audio_arg
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(f"No se encontró el audio: {audio_arg}")

def load_state(state_path: str) -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state_path: str, state: Dict[str, Any]) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)

def normalize_text_for_similarity(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sáéíóúñü]", "", s)
    return s

def similarity(a: str, b: str) -> float:
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
    if not segments:
        return []
    out: List[Dict[str, Any]] = []
    last_texts: List[str] = []
    repeat_counter: Dict[str, int] = {}
    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        is_dup = False
        for prev in last_texts[-DEDUP_WINDOW:]:
            if similarity(txt, prev) >= SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if is_dup:
            key = normalize_text_for_similarity(txt)
            repeat_counter[key] = repeat_counter.get(key, 0) + 1
            if repeat_counter[key] >= MAX_REPEAT_PHRASE:
                continue
            continue
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

def load_replacements(path: str) -> List[Tuple[re.Pattern, str]]:
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
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([¿¡])\s+", r"\1", t)
    return t.strip()

def apply_replacements(text: str, reps: List[Tuple[re.Pattern, str]]) -> str:
    t = text
    for pat, repl in reps:
        t = pat.sub(repl, t)
    return t

def remove_fillers(text: str) -> str:
    t = text
    for pat in FILLER_PATTERNS:
        t = re.sub(pat, "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t

def _should_merge(prev_text: str, gap_s: float, merge_gap_s: float) -> bool:
    if gap_s > merge_gap_s:
        return False
    if prev_text.strip().endswith((".", "!", "?", "…")):
        return False
    return True

def postprocess_segments(segments: List[Dict[str, Any]], reps: List[Tuple[re.Pattern, str]], do_remove_fillers: bool, merge_gap_s: float) -> List[Dict[str, Any]]:
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

def speaker_label(i: int) -> str:
    if 0 <= i < 26:
        return f"Participante {chr(ord('A') + i)}"
    return f"Participante S{i+1}"

def diarize_light(segments: List[Dict[str, Any]], num_speakers: int, turn_gap_s: float, force_turn_max_s: float) -> List[Dict[str, Any]]:
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
    if not segments:
        return segments
    num_speakers = max(1, int(num_speakers))
    max_key = min(num_speakers, 9)
    print("\n=== Revisión de diarización (rápida) ===")
    print(f"Enter=mantener | 1..{max_key}=reasignar | b=atrás | q=salir\n")
    i = 0
    out = [dict(s) for s in segments]
    while i < len(out):
        s = out[i]
        sp = s.get("speaker","Participante A")
        print(f"[{i+1}/{len(out)}] {s['start']:.2f}s -> {s['end']:.2f}s | {sp}")
        print(f"  {s.get('text','')}\n")
        cmd = input(">> ").strip().lower()
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
        print("Comando no válido.\n")
    print("Revisión finalizada.\n")
    return out

def transcribe_one_chunk(model: WhisperModel, chunk_path: str, language: str, beam_size: int, word_timestamps: bool) -> List[Dict[str, Any]]:
    segments_iter, _ = model.transcribe(chunk_path, language=language, beam_size=beam_size, word_timestamps=word_timestamps)
    out: List[Dict[str, Any]] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            out.append({"start_local": float(seg.start), "end_local": float(seg.end), "text": text})
    return out

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

def prompt(msg: str, default: Optional[str]=None) -> str:
    q = f"{msg} [{default}]: " if default is not None else f"{msg}: "
    s = input(q).strip()
    return s if s else (default if default is not None else "")

def prompt_int(msg: str, default: int, min_v: int=1, max_v: int=10000) -> int:
    while True:
        try:
            v = int(prompt(msg, str(default)))
            if v < min_v or v > max_v:
                print(f"  ↳ Ingresa un valor entre {min_v} y {max_v}."); continue
            return v
        except ValueError:
            print("  ↳ Ingresa un entero válido.")

def prompt_float(msg: str, default: float, min_v: float=0.0, max_v: float=10000.0) -> float:
    while True:
        try:
            v = float(prompt(msg, str(default)))
            if v < min_v or v > max_v:
                print(f"  ↳ Ingresa un valor entre {min_v} y {max_v}."); continue
            return v
        except ValueError:
            print("  ↳ Ingresa un número válido.")

def assisted_args() -> argparse.Namespace:
    print("\n=== Transcriptor Simple v2 (modo asistido) ===\n")
    while True:
        a = prompt("Ruta del audio (arrastrar/pegar la ruta)", "")
        if not a:
            print("  ↳ Necesito una ruta."); continue
        try:
            a = resolve_audio_path(a); break
        except Exception as e:
            print(f"  ↳ {e}")

    model = prompt("Modelo (small/medium)", DEFAULT_MODEL).lower()
    if model not in {"small","medium"}:
        model = DEFAULT_MODEL

    fallback = prompt("Fallback (small/none)", DEFAULT_FALLBACK_MODEL).lower()
    if fallback == "none":
        fallback = ""
    elif fallback not in {"small","medium"}:
        fallback = DEFAULT_FALLBACK_MODEL

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
            print("  ↳ No encontré ese JSON; continúo sin replacements externos.")
            replacements_json = ""
        write_clean = prompt("Escribir salida limpia adicional (y/n)", "y").lower().startswith("y")

    diarize = prompt("Diarización ligera (Participante A/B/...) (y/n)", "y").lower().startswith("y")
    num_speakers = DEFAULT_NUM_SPEAKERS
    turn_gap_s = DEFAULT_TURN_GAP_S
    force_turn_max_s = DEFAULT_FORCE_TURN_MAX_S
    review_diar = DEFAULT_REVIEW_DIARIZATION
    if diarize:
        num_speakers = prompt_int("Número de participantes (2 recomendado)", DEFAULT_NUM_SPEAKERS, 1, 9)
        turn_gap_s = prompt_float("Cambio de turno si pausa ≥ (s)", DEFAULT_TURN_GAP_S, 0.1, 10.0)
        force_turn_max_s = prompt_float("Forzar cambio si bloque ≥ (s)", DEFAULT_FORCE_TURN_MAX_S, 5.0, 600.0)
        review_diar = prompt("Revisar diarización al final (y/n)", "n").lower().startswith("y")

    workdir = prompt("Carpeta work", DEFAULT_WORKDIR)
    outdir = prompt("Carpeta salida", DEFAULT_OUTDIR)
    logdir = prompt("Carpeta logs", DEFAULT_LOGDIR)

    return argparse.Namespace(
        audio=a, workdir=workdir, outdir=outdir, logdir=logdir,
        model=model, fallback_model=fallback, compute_type=DEFAULT_COMPUTE_TYPE,
        language=DEFAULT_LANGUAGE, chunk_s=chunk_s, overlap_s=overlap_s,
        beam=beam, word_timestamps=False, normalize=normalize, resume=resume,
        postprocess=post, replacements_json=replacements_json, merge_gap_s=merge_gap_s,
        remove_fillers=remove_fillers, write_clean=write_clean,
        diarize=diarize, num_speakers=num_speakers, turn_gap_s=turn_gap_s,
        force_turn_max_s=force_turn_max_s, review_diarization=review_diar,
        keep_chunks=False
    )

def validate_args(args: argparse.Namespace) -> None:
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
    if args.replacements_json and not os.path.exists(args.replacements_json):
        raise ValueError("replacements_json no existe")
    if args.diarize and (args.num_speakers < 1 or args.num_speakers > 9):
        raise ValueError("num_speakers debe estar entre 1 y 9")

def run_pipeline(args: argparse.Namespace) -> None:
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
    write_log(log_path, f"🧠 Modelo: {args.model} | Fallback: {args.fallback_model or '(none)'} | compute: {args.compute_type}")
    write_log(log_path, f"⚙️ chunk={args.chunk_s}s overlap={args.overlap_s}s beam={args.beam} normalize={bool(args.normalize)} resume={bool(args.resume)}")
    write_log(log_path, f"🧼 postprocess={bool(args.postprocess)} remove_fillers={bool(args.remove_fillers)} merge_gap={args.merge_gap_s}s replacements={args.replacements_json or '(default)'}")
    write_log(log_path, f"🗣️ diarize={bool(args.diarize)} num_speakers={getattr(args,'num_speakers',0)} turn_gap={getattr(args,'turn_gap_s',0)} force_turn_max={getattr(args,'force_turn_max_s',0)} review={getattr(args,'review_diarization',False)}")

    state = load_state(state_path) if args.resume else {}
    if not state:
        state = {
            "audio_path": audio_path, "audio_name": audio_name,
            "model": args.model, "fallback_model": args.fallback_model,
            "compute_type": args.compute_type, "language": args.language,
            "chunk_s": args.chunk_s, "overlap_s": args.overlap_s, "beam": args.beam,
            "normalize": bool(args.normalize),
            "completed_chunks": [], "failed_chunks": [], "total_chunks": None,
            "partials_jsonl": partials_jsonl,
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
    existing = sorted([p for p in Path(chunks_dir).glob(f"{audio_name}_chunk_*.wav")])
    if existing:
        chunks: List[Chunk] = []
        for p in existing:
            m = re.search(r"_chunk_(\d{4})\.wav$", p.name)
            if not m:
                continue
            idx = int(m.group(1))
            d = ffprobe_duration_seconds(str(p))
            start = idx * (args.chunk_s - args.overlap_s)
            chunks.append(Chunk(idx=idx, path=str(p), start_s=start, end_s=start+d))
        chunks.sort(key=lambda c: c.idx)
        write_log(log_path, f"♻️ Reutilizando {len(chunks)} chunks existentes.")
    else:
        write_log(log_path, "✂️ Generando chunks...")
        chunks = split_audio_fixed(input_for_split, chunks_dir, args.chunk_s, args.overlap_s, audio_name)
        write_log(log_path, f"✅ Chunks generados: {len(chunks)}")

    state["total_chunks"] = len(chunks)
    save_state(state_path, state)

    write_log(log_path, f"🧠 Cargando modelo principal: {args.model} ({args.compute_type})")
    model_main = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)

    model_fallback = None
    if args.fallback_model and args.fallback_model != args.model:
        write_log(log_path, f"🧠 Cargando fallback: {args.fallback_model} ({args.compute_type})")
        model_fallback = WhisperModel(args.fallback_model, device="cpu", compute_type=args.compute_type)

    completed = set(state.get("completed_chunks", []))
    failed = set(state.get("failed_chunks", []))
    if not os.path.exists(partials_jsonl):
        open(partials_jsonl, "w", encoding="utf-8").close()

    t0 = time.time()
    for ch in chunks:
        if ch.idx in completed:
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
        "compute_type": args.compute_type, "language": args.language,
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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transcriptor Simple v2 (modo asistido si no pasas audio).")
    p.add_argument("audio", nargs="?", default=None, help="Ruta del audio. Si omites, entra en asistente.")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--logdir", default=DEFAULT_LOGDIR)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["small","medium"])
    p.add_argument("--fallback_model", default=DEFAULT_FALLBACK_MODEL, choices=["","small","medium"])
    p.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE)
    p.add_argument("--language", default=DEFAULT_LANGUAGE)
    p.add_argument("--chunk_s", type=int, default=DEFAULT_CHUNK_S)
    p.add_argument("--overlap_s", type=float, default=DEFAULT_OVERLAP_S)
    p.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    p.add_argument("--word_timestamps", action="store_true", default=DEFAULT_WORD_TIMESTAMPS)
    p.add_argument("--normalize", action="store_true", default=DEFAULT_NORMALIZE)
    p.add_argument("--resume", action="store_true", default=True)

    p.add_argument("--postprocess", action="store_true", default=DEFAULT_POSTPROCESS)
    p.add_argument("--replacements_json", default="")
    p.add_argument("--merge_gap_s", type=float, default=DEFAULT_MERGE_GAP_S)
    p.add_argument("--remove_fillers", action="store_true", default=DEFAULT_REMOVE_FILLERS)
    p.add_argument("--write_clean", action="store_true", default=DEFAULT_WRITE_CLEAN)

    p.add_argument("--diarize", action="store_true", default=DEFAULT_DIARIZE)
    p.add_argument("--num_speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    p.add_argument("--turn_gap_s", type=float, default=DEFAULT_TURN_GAP_S)
    p.add_argument("--force_turn_max_s", type=float, default=DEFAULT_FORCE_TURN_MAX_S)
    p.add_argument("--review_diarization", action="store_true", default=DEFAULT_REVIEW_DIARIZATION)
    return p

def main() -> None:
    args = build_parser().parse_args()
    if args.audio is None:
        args = assisted_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
