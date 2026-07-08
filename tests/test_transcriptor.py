import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_utils import split_audio_fixed
from src.diarization_utils import diarize_light, review_diarization_interactive, speaker_label
from src.state_utils import config_signature, load_state, resolve_audio_path, save_state
from src.system_utils import probe_audio_info
from src.transcriptor import (
    audio_needs_normalization,
    format_duration,
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

    def test_format_duration_is_human_readable(self) -> None:
        self.assertEqual(format_duration(0), "0m 00s")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3661), "1h 01m 01s")

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
            diarize=False,
            num_speakers=2,
            turn_gap_s=1.2,
            force_turn_max_s=30.0,
        )
        with self.assertRaisesRegex(ValueError, "audio no puede estar vacío"):
            validate_args(args)

    def test_prompt_bool_reprompts_until_valid(self) -> None:
        answers = iter(["quizas", "s"])
        with patch("src.transcriptor.prompt", side_effect=lambda msg, default=None: next(answers)):
            self.assertTrue(prompt_bool("Confirmar", True))


class StateUtilsTests(unittest.TestCase):
    def test_resolve_audio_path_strips_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_text("fake", encoding="utf-8")
            resolved = resolve_audio_path(f'"{audio}"')
            self.assertEqual(resolved, str(audio.resolve()))

    def test_save_and_load_state_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            payload = {"completed_chunks": [1, 2], "config_sig": "abc"}
            save_state(str(state_path), payload)
            self.assertEqual(load_state(str(state_path)), payload)

    def test_load_state_returns_empty_dict_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text("{invalid", encoding="utf-8")
            self.assertEqual(load_state(str(state_path)), {})

    def test_config_signature_is_stable_for_same_content(self) -> None:
        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}
        self.assertEqual(config_signature(a), config_signature(b))


class AudioUtilsTests(unittest.TestCase):
    def test_probe_audio_info_includes_duration(self) -> None:
        payload = '{"streams":[{"sample_rate":"16000","channels":1}],"format":{"duration":"12.5"}}'
        with patch("src.system_utils.run_cmd", return_value=payload):
            info = probe_audio_info("audio.wav")
        self.assertEqual(info["sample_rate"], 16000)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["duration_s"], 12.5)

    def test_split_audio_fixed_skips_tail_already_covered_by_overlap(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.audio_utils.run_cmd", side_effect=lambda cmd: calls.append(cmd) or ""):
                chunks = split_audio_fixed("audio.wav", tmpdir, chunk_s=45, overlap_s=1.0, prefix="audio", duration_s=45.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(chunks[0].start_s, 0.0)
        self.assertEqual(chunks[0].end_s, 45.0)

    def test_split_audio_fixed_keeps_tail_with_uncovered_audio(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.audio_utils.run_cmd", side_effect=lambda cmd: calls.append(cmd) or ""):
                chunks = split_audio_fixed("audio.wav", tmpdir, chunk_s=45, overlap_s=1.0, prefix="audio", duration_s=45.5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(chunks[1].start_s, 44.0)
        self.assertEqual(chunks[1].end_s, 45.5)


class DiarizationUtilsTests(unittest.TestCase):
    def test_speaker_label_supports_letters_and_overflow(self) -> None:
        self.assertEqual(speaker_label(0), "Participante A")
        self.assertEqual(speaker_label(26), "Participante S27")

    def test_diarize_light_switches_speaker_on_gap(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hola"},
            {"start": 3.0, "end": 4.0, "text": "mundo"},
        ]
        diarized = diarize_light(segments, num_speakers=2, turn_gap_s=1.0, force_turn_max_s=30.0)
        self.assertEqual(diarized[0]["speaker"], "Participante A")
        self.assertEqual(diarized[0]["speaker_index"], 0)
        self.assertEqual(diarized[0]["speaker_turn_index"], 0)
        self.assertEqual(diarized[0]["diarization_reason"], "start")
        self.assertEqual(diarized[1]["speaker"], "Participante B")
        self.assertEqual(diarized[1]["speaker_index"], 1)
        self.assertEqual(diarized[1]["speaker_turn_index"], 1)
        self.assertEqual(diarized[1]["diarization_reason"], "gap")
        self.assertGreaterEqual(diarized[1]["diarization_confidence"], 0.8)

    def test_diarize_light_holds_short_interjection_after_gap(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "text": "estamos revisando el punto principal"},
            {"start": 4.0, "end": 4.4, "text": "sí"},
        ]
        diarized = diarize_light(segments, num_speakers=2, turn_gap_s=1.0, force_turn_max_s=30.0)
        self.assertEqual(diarized[1]["speaker"], "Participante A")
        self.assertEqual(diarized[1]["speaker_turn_index"], 0)
        self.assertEqual(diarized[1]["diarization_reason"], "short_segment_hold")
        self.assertLess(diarized[1]["diarization_confidence"], 0.8)

    def test_diarize_light_switches_long_segment_after_gap(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "text": "estamos revisando el punto principal"},
            {"start": 4.0, "end": 5.2, "text": "quiero responder con contexto"},
        ]
        diarized = diarize_light(segments, num_speakers=2, turn_gap_s=1.0, force_turn_max_s=30.0)
        self.assertEqual(diarized[1]["speaker"], "Participante B")
        self.assertEqual(diarized[1]["speaker_turn_index"], 1)
        self.assertEqual(diarized[1]["diarization_reason"], "gap")

    def test_diarize_light_switches_speaker_on_long_turn(self) -> None:
        segments = [
            {"start": 0.0, "end": 3.0, "text": "uno"},
            {"start": 3.1, "end": 6.2, "text": "dos"},
        ]
        diarized = diarize_light(segments, num_speakers=2, turn_gap_s=10.0, force_turn_max_s=2.0)
        self.assertEqual(diarized[1]["speaker"], "Participante B")
        self.assertEqual(diarized[1]["diarization_reason"], "max_turn")
        self.assertLess(diarized[1]["diarization_confidence"], 0.8)

    def test_diarize_light_sorts_segments_by_time(self) -> None:
        segments = [
            {"start": 3.0, "end": 4.0, "text": "segundo"},
            {"start": 0.0, "end": 1.0, "text": "primero"},
        ]
        diarized = diarize_light(segments, num_speakers=2, turn_gap_s=1.0, force_turn_max_s=30.0)
        self.assertEqual([s["text"] for s in diarized], ["primero", "segundo"])
        self.assertEqual(diarized[1]["diarization_reason"], "gap")

    def test_review_diarization_interactive_reassigns_speaker(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "hola", "speaker": "Participante A"}]
        outputs = []
        answers = iter(["2", "q"])
        with patch("src.diarization_utils.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = True
            reviewed = review_diarization_interactive(
                segments,
                num_speakers=2,
                printer=outputs.append,
                input_func=lambda _: next(answers),
            )
        self.assertEqual(reviewed[0]["speaker"], "Participante B")
        self.assertEqual(reviewed[0]["speaker_index"], 1)
        self.assertTrue(any("Revisión de diarización" in line for line in outputs))


if __name__ == "__main__":
    unittest.main()
