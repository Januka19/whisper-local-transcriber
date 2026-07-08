from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def ensure_ffmpeg(printer: Optional[Callable[[str], None]] = None) -> None:
    emit = printer or print
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        emit("❌ No se encontró ffmpeg/ffprobe en PATH.")
        emit("   En Fedora: sudo dnf install -y ffmpeg-free ffmpeg-free-devel")
        raise SystemExit(1)


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Comando falló: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return (p.stdout or "").strip()


def ffprobe_duration_seconds(path: str) -> float:
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(out)


def safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def probe_audio_info(path: str) -> Dict[str, Any]:
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels:format=duration",
        "-of", "json",
        path,
    ])
    empty = {"sample_rate": 0, "channels": 0, "duration_s": 0.0}
    try:
        data = json.loads(out)
    except Exception:
        return empty

    streams = data.get("streams") or []
    if not streams:
        return empty

    stream = streams[0] or {}
    fmt = data.get("format") or {}
    return {
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration_s": _safe_float(fmt.get("duration")),
    }


def audio_needs_normalization(audio_info: Dict[str, Any]) -> bool:
    sample_rate = int(audio_info.get("sample_rate") or 0)
    channels = int(audio_info.get("channels") or 0)
    return sample_rate != 16000 or channels != 1


class Logger:
    """Logger simple: archivo + consola, con timestamps consistentes."""

    def __init__(self, log_path: str, printer: Optional[Callable[[str], None]] = None):
        self.log_path = log_path
        self.printer = printer or print
        safe_mkdir(str(Path(log_path).parent))

    def write(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
        self.printer(msg)
