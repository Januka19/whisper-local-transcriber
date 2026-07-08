from __future__ import annotations

import hashlib
import json
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
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state_path: str, state: Dict[str, Any]) -> None:
    path = Path(state_path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def config_signature(d: Dict[str, Any]) -> str:
    relevant = json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(relevant).hexdigest()
