import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcriptor import (
    audio_needs_normalization,
    normalize_compute_type,
    normalize_language,
    prompt_bool,
    validate_args,
)


class TranscriptorValidationTests(unittest.TestCase):
    def test_audio_needs_normalization_detects_non_standard_audio(self) -> None:
        self.assertTrue(audio_needs_normalization({"sample_rate": 44100, "channels": 2}))

    def test_audio_needs_normalization_accepts_16k_mono(self) -> None:
        self.assertFalse(audio_needs_normalization({"sample_rate": 16000, "channels": 1}))

    def test_normalize_compute_type_rejects_invalid_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "compute_type inválido"):
            normalize_compute_type("gpu-fast")

    def test_normalize_language_rejects_invalid_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "language inválido"):
            normalize_language("spanish")

    def test_validate_args_rejects_blank_audio(self) -> None:
        args = argparse.Namespace(
            audio="   ",
            workdir="work",
            outdir="salida",
            logdir="logs",
            model="medium",
            compute_type="int8",
            fallback_model="",
            fallback_compute_type="int8",
            cpu_threads=0,
            num_workers=1,
            language="es",
            chunk_s=45,
            overlap_s=0.4,
            beam=1,
            replacements_json="",
            diarize=False,
            num_speakers=2,
            merge_gap_s=0.8,
            turn_gap_s=1.2,
            force_turn_max_s=30.0,
        )
        with self.assertRaisesRegex(ValueError, "audio no puede estar vacío"):
            validate_args(args)

    def test_prompt_bool_reprompts_until_valid(self) -> None:
        answers = iter(["quizas", "s"])
        with patch("src.transcriptor.prompt", side_effect=lambda msg, default=None: next(answers)):
            self.assertTrue(prompt_bool("Confirmar", True))


if __name__ == "__main__":
    unittest.main()
