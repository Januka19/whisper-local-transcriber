from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


def resolve_audio_path(audio_arg: str) -> str:
    if not audio_arg:
        raise FileNotFoundError("Ruta de audio vacía.")
    cleaned = audio_arg.strip().strip("\'\"")
    p = Path(cleaned).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"No se encontró el audio: {p}")
    return str(p)


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


def config_signature(d: Dict[str, Any]) -> str:
    relevant = json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(relevant).hexdigest()
