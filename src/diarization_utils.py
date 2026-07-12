from __future__ import annotations

import importlib
import importlib.util
import tarfile
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


DEFAULT_DIARIZATION_MODEL_DIR = "models/diarization"
SHERPA_SEGMENTATION_FILENAME = "pyannote-segmentation-3.0-int8.onnx"
SHERPA_EMBEDDING_FILENAME = "nemo-titanet-small.onnx"
MIN_SHERPA_MODEL_BYTES = 1_000_000
SHERPA_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SHERPA_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_small.onnx"
)


def speaker_label(i: int) -> str:
    if 0 <= i < 26:
        return f"Participante {chr(ord('A') + i)}"
    return f"Participante S{i+1}"


def sherpa_model_paths(model_dir: str) -> Dict[str, Path]:
    root = Path(model_dir).expanduser()
    return {
        "segmentation": root / SHERPA_SEGMENTATION_FILENAME,
        "embedding": root / SHERPA_EMBEDDING_FILENAME,
    }


def _model_file_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_SHERPA_MODEL_BYTES


def sherpa_diarization_ready(model_dir: str) -> tuple[bool, str]:
    try:
        dependency_available = importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):
        dependency_available = False
    if not dependency_available:
        return False, "falta la dependencia sherpa-onnx"
    invalid = [
        path.name
        for path in sherpa_model_paths(model_dir).values()
        if not _model_file_complete(path)
    ]
    if invalid:
        return False, "faltan pesos locales o están incompletos: " + ", ".join(invalid)
    return True, "modelo Sherpa ONNX disponible"


def require_sherpa_diarization(model_dir: str) -> str:
    ready, reason = sherpa_diarization_ready(model_dir)
    if not ready:
        raise RuntimeError(
            f"No se puede usar la diarización Sherpa: {reason}. "
            "Ejecuta --setup_diarization_models y vuelve a intentar."
        )
    return reason


def create_sherpa_diarizer(
    model_dir: str,
    num_speakers: int,
    num_threads: int = 1,
) -> Any:
    """Load and validate Sherpa models before an expensive transcription starts."""
    ready, reason = sherpa_diarization_ready(model_dir)
    if not ready:
        raise RuntimeError(f"Diarización Sherpa no disponible: {reason}")

    sherpa_onnx = importlib.import_module("sherpa_onnx")
    paths = sherpa_model_paths(model_dir)
    threads = max(1, int(num_threads or 1))
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(paths["segmentation"])
            ),
            num_threads=threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(paths["embedding"]),
            num_threads=threads,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=max(1, int(num_speakers)),
            threshold=0.5,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("La configuración de diarización Sherpa no es válida.")
    try:
        return sherpa_onnx.OfflineSpeakerDiarization(config)
    except Exception as exc:
        raise RuntimeError(f"No se pudieron cargar los modelos de diarización Sherpa: {exc}") from exc


def _download_file(url: str, destination: Path, printer: Callable[[str], None]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    printer(f"Descargando {destination.name}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def setup_sherpa_diarization_models(
    model_dir: str,
    printer: Callable[[str], None] = print,
) -> Dict[str, str]:
    """Download the two official Sherpa model files after explicit user action."""
    paths = sherpa_model_paths(model_dir)
    root = paths["segmentation"].parent
    root.mkdir(parents=True, exist_ok=True)

    if not _model_file_complete(paths["segmentation"]):
        archive = root / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        _download_file(SHERPA_SEGMENTATION_URL, archive, printer)
        try:
            with tarfile.open(archive, "r:bz2") as bundle:
                member = next(
                    (
                        item
                        for item in bundle.getmembers()
                        if item.isfile() and Path(item.name).name == "model.int8.onnx"
                    ),
                    None,
                )
                if member is None:
                    raise RuntimeError("El paquete de segmentación no contiene model.int8.onnx.")
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError("No se pudo leer el modelo de segmentación descargado.")
                temporary = paths["segmentation"].with_name(paths["segmentation"].name + ".part")
                with source, temporary.open("wb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                temporary.replace(paths["segmentation"])
        finally:
            archive.unlink(missing_ok=True)
        if not _model_file_complete(paths["segmentation"]):
            paths["segmentation"].unlink(missing_ok=True)
            raise RuntimeError("El modelo de segmentación descargado está incompleto.")
    else:
        printer(f"Reutilizando {paths['segmentation']}")

    if not _model_file_complete(paths["embedding"]):
        _download_file(SHERPA_EMBEDDING_URL, paths["embedding"], printer)
        if not _model_file_complete(paths["embedding"]):
            paths["embedding"].unlink(missing_ok=True)
            raise RuntimeError("El modelo de embeddings descargado está incompleto.")
    else:
        printer(f"Reutilizando {paths['embedding']}")

    return {name: str(path) for name, path in paths.items()}


def assign_speakers_from_turns(
    segments: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not segments or not turns:
        return [dict(segment) for segment in segments]

    speaker_ids = sorted({int(turn["speaker_index"]) for turn in turns})
    speaker_map = {speaker_id: index for index, speaker_id in enumerate(speaker_ids)}
    ordered_turns = sorted(turns, key=lambda turn: (float(turn["start"]), float(turn["end"])))
    out: List[Dict[str, Any]] = []
    previous_speaker: Optional[int] = None
    turn_index = -1
    turn_cursor = 0

    for segment in sorted(segments, key=lambda item: (float(item["start"]), float(item["end"]))):
        start = float(segment["start"])
        end = float(segment["end"])
        duration = max(0.001, end - start)
        overlap_by_speaker: Dict[int, float] = {}

        while (
            turn_cursor < len(ordered_turns)
            and float(ordered_turns[turn_cursor]["end"]) <= start
        ):
            turn_cursor += 1

        candidate_index = turn_cursor
        while candidate_index < len(ordered_turns):
            turn = ordered_turns[candidate_index]
            if float(turn["start"]) >= end:
                break
            overlap = max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))
            if overlap > 0:
                speaker_id = int(turn["speaker_index"])
                overlap_by_speaker[speaker_id] = overlap_by_speaker.get(speaker_id, 0.0) + overlap
            candidate_index += 1

        if overlap_by_speaker:
            raw_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda item: item[1])
            confidence = min(1.0, best_overlap / duration)
            reason = "model_overlap"
        else:
            enriched = dict(segment)
            enriched["speaker"] = None
            enriched["speaker_index"] = None
            enriched["speaker_turn_index"] = None
            enriched["diarization_reason"] = "model_no_overlap"
            enriched["diarization_confidence"] = 0.0
            out.append(enriched)
            continue

        speaker_index = speaker_map[raw_speaker]
        if speaker_index != previous_speaker:
            turn_index += 1
            previous_speaker = speaker_index

        enriched = dict(segment)
        enriched["speaker"] = speaker_label(speaker_index)
        enriched["speaker_index"] = speaker_index
        enriched["speaker_turn_index"] = turn_index
        enriched["diarization_reason"] = reason
        enriched["diarization_confidence"] = round(confidence, 4)
        out.append(enriched)

    return out


def diarize_sherpa(
    audio_wav: str,
    segments: List[Dict[str, Any]],
    num_speakers: int,
    model_dir: str,
    num_threads: int = 1,
    diarizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run the lightweight local Sherpa ONNX model and align turns to text."""
    if not segments:
        return []
    numpy = importlib.import_module("numpy")

    with wave.open(audio_wav, "rb") as wav_file:
        if wav_file.getframerate() != 16000 or wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise RuntimeError("Sherpa requiere WAV PCM de 16 kHz, mono y 16 bits.")
        samples = numpy.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
        samples = samples.astype("float32") / 32768.0

    if diarizer is None:
        diarizer = create_sherpa_diarizer(model_dir, num_speakers, num_threads)
    result = diarizer.process(samples).sort_by_start_time()
    turns = [
        {"start": float(item.start), "end": float(item.end), "speaker_index": int(item.speaker)}
        for item in result
    ]
    if not turns:
        raise RuntimeError("El modelo de diarización no detectó turnos de voz.")
    return assign_speakers_from_turns(segments, turns)


def review_diarization_interactive(
    segments: List[Dict[str, Any]],
    num_speakers: int,
    printer: Callable[[str], None],
    input_func: Callable[[str], str] = input,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    try:
        if not sys.stdin or not sys.stdin.isatty():
            printer("⚠️ Revisión solicitada pero no hay terminal interactivo; se omite.")
            return segments
    except Exception:
        printer("⚠️ No pude validar TTY; se omite revisión para evitar bloqueo.")
        return segments

    num_speakers = max(1, int(num_speakers))
    max_key = min(num_speakers, 9)

    printer("\n=== Revisión de diarización (rápida) ===")
    printer(f"Enter=mantener | 1..{max_key}=reasignar | b=atrás | q=salir\n")

    out = [dict(s) for s in segments]
    i = 0
    while i < len(out):
        s = out[i]
        sp = s.get("speaker", "Participante A")
        printer(f"[{i+1}/{len(out)}] {s['start']:.2f}s -> {s['end']:.2f}s | {sp}")
        printer(f"  {s.get('text', '')}\n")
        try:
            cmd = input_func(">> ").strip().lower()
        except EOFError:
            printer("\n⚠️ EOF — saliendo de revisión.")
            break

        if cmd == "":
            i += 1
            continue
        if cmd == "q":
            break
        if cmd == "b":
            i = max(0, i - 1)
            continue
        if cmd.isdigit():
            k = int(cmd)
            if 1 <= k <= num_speakers:
                out[i]["speaker"] = speaker_label(k - 1)
                out[i]["speaker_index"] = k - 1
                i += 1
                continue

        printer("Comando no válido.\n")

    printer("Revisión finalizada.\n")
    return out
