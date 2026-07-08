from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from src.system_utils import ffprobe_duration_seconds, run_cmd, safe_mkdir
except ModuleNotFoundError:
    from system_utils import ffprobe_duration_seconds, run_cmd, safe_mkdir


@dataclass(frozen=True)
class Chunk:
    idx: int
    path: str
    start_s: float
    end_s: float


def normalize_to_wav_16k_mono(input_audio: str, out_wav: str) -> None:
    run_cmd([
        "ffmpeg", "-y", "-i", input_audio,
        "-ac", "1", "-ar", "16000", "-vn",
        "-af", "highpass=f=80,volume=1.2",
        out_wav,
    ])


def split_audio_fixed(
    input_audio: str,
    chunks_dir: str,
    chunk_s: int,
    overlap_s: float,
    prefix: str,
    duration_s: Optional[float] = None,
) -> List[Chunk]:
    safe_mkdir(chunks_dir)
    dur = float(duration_s) if duration_s and duration_s > 0 else ffprobe_duration_seconds(input_audio)
    if dur <= 0:
        return []

    chunk_len = float(chunk_s)
    overlap = float(overlap_s)
    step = chunk_len - overlap
    if step <= 0:
        raise ValueError("chunk_s debe ser mayor que overlap_s.")

    chunks: List[Chunk] = []
    idx = 0
    t = 0.0

    while t < dur:
        remaining = dur - t
        if chunks and overlap > 0 and remaining <= overlap:
            break

        start = max(0.0, t)
        end = min(dur, t + chunk_len)
        duration = end - start

        out_path = str(Path(chunks_dir) / f"{prefix}_chunk_{idx:04d}.wav")
        run_cmd([
            "ffmpeg", "-y",
            "-ss", str(start), "-i", input_audio,
            "-t", str(duration),
            "-ac", "1", "-ar", "16000", "-vn",
            out_path,
        ])

        chunks.append(Chunk(idx=idx, path=out_path, start_s=start, end_s=end))
        idx += 1
        t += step

    return chunks
