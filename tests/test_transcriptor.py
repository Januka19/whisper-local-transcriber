import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcriptor import audio_needs_normalization


def test_audio_needs_normalization_detects_non_standard_audio() -> None:
    assert audio_needs_normalization({"sample_rate": 44100, "channels": 2}) is True


def test_audio_needs_normalization_accepts_16k_mono() -> None:
    assert audio_needs_normalization({"sample_rate": 16000, "channels": 1}) is False
