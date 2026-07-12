import argparse
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_utils import Chunk, split_audio_fixed
from src.diarization_utils import (
    assign_speakers_from_turns,
    create_sherpa_diarizer,
    require_sherpa_diarization,
    review_diarization_interactive,
    setup_sherpa_diarization_models,
    sherpa_model_paths,
    speaker_label,
)
from src.state_utils import audio_source_identity, config_signature, load_state, resolve_audio_path, save_state
from src.system_utils import probe_audio_info
from src.transcriptor import (
    audio_needs_normalization,
    canonical_model_id,
    format_duration,
    normalize_compute_type,
    normalize_language,
    prompt_bool,
    run_pipeline,
    validate_args,
)


def pipeline_args(audio: Path, root: Path, resume: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        audio=str(audio),
        workdir=str(root / "work"),
        outdir=str(root / "salida"),
        logdir=str(root / "logs"),
        model="medium",
        fallback_model="",
        device="cpu",
        compute_type="int8",
        fallback_compute_type="int8",
        cpu_threads=0,
        num_workers=1,
        language="es",
        chunk_s=45,
        overlap_s=0.4,
        beam=1,
        word_timestamps=False,
        normalize=False,
        resume=resume,
        vad_filter=True,
        diarize=False,
        num_speakers=2,
        review_diarization=False,
        diarization_model_dir=str(root / "models" / "diarization"),
        force_reuse_chunks=False,
        clean=False,
    )


class TranscriptorValidationTests(unittest.TestCase):
    def test_turbo_aliases_use_upstream_faster_whisper_model(self) -> None:
        self.assertEqual(canonical_model_id("turbo-int8"), "large-v3-turbo")
        self.assertEqual(canonical_model_id("large-v3-turbo"), "large-v3-turbo")

    def test_quality_alias_uses_full_large_v3(self) -> None:
        self.assertEqual(canonical_model_id("quality"), "large-v3")

    def test_legacy_turbo_alias_remains_available_offline(self) -> None:
        self.assertEqual(
            canonical_model_id("turbo-int8-legacy"),
            "Zoont/faster-whisper-large-v3-turbo-int8-ct2",
        )

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

    def test_load_state_returns_empty_dict_for_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text("[]", encoding="utf-8")
            self.assertEqual(load_state(str(state_path)), {})

    def test_config_signature_is_stable_for_same_content(self) -> None:
        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}
        self.assertEqual(config_signature(a), config_signature(b))

    def test_audio_source_identity_changes_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "audio.wav"
            audio.write_bytes(b"first")
            first = audio_source_identity(str(audio))
            audio.write_bytes(b"a different payload")
            second = audio_source_identity(str(audio))
        self.assertEqual(first["path"], second["path"])
        self.assertNotEqual(first["size"], second["size"])


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


class PipelineRegressionTests(unittest.TestCase):
    @staticmethod
    def _fake_split(input_audio, chunks_dir, chunk_s, overlap_s, prefix, duration_s):
        chunk_path = Path(chunks_dir) / f"{prefix}_chunk_0000.wav"
        chunk_path.write_bytes(b"chunk")
        return [Chunk(idx=0, path=str(chunk_path), start_s=0.0, end_s=1.0)]

    def test_no_resume_discards_stale_partials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            workdir = root / "work"
            workdir.mkdir()
            partials = workdir / "audio_partials.jsonl"
            partials.write_text('{"text":"resultado antiguo"}\n', encoding="utf-8")
            args = pipeline_args(audio, root, resume=False)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch("src.transcriptor.transcribe_one_chunk", return_value=[{"start_local": 0.0, "end_local": 1.0, "text": "resultado nuevo"}]):
                run_pipeline(args)

            lines = [json.loads(line) for line in partials.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["text"] for line in lines], ["resultado nuevo"])

    def test_successful_retry_removes_chunk_from_failed_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch(
                        "src.transcriptor.transcribe_one_chunk",
                        side_effect=[RuntimeError("fallo temporal"), [{"start_local": 0.0, "end_local": 1.0, "text": "recuperado"}]],
                    ):
                run_pipeline(args)
                run_pipeline(args)

            state = load_state(str(root / "work" / "audio_estado.json"))
            self.assertEqual(state["completed_chunks"], [0])
            self.assertEqual(state["failed_chunks"], [])

    def test_force_reuse_chunks_never_reuses_incompatible_partials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch(
                        "src.transcriptor.transcribe_one_chunk",
                        side_effect=[
                            [{"start_local": 0.0, "end_local": 1.0, "text": "resultado en español"}],
                            [{"start_local": 0.0, "end_local": 1.0, "text": "english result"}],
                        ],
                    ):
                run_pipeline(args)
                args.language = "en"
                args.force_reuse_chunks = True
                run_pipeline(args)

            partials = root / "work" / "audio_partials.jsonl"
            lines = [json.loads(line) for line in partials.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["text"] for line in lines], ["english result"])

    def test_sherpa_backend_is_used_and_recorded_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)
            args.diarize = True

            model_result = [{
                "start": 0.0,
                "end": 1.0,
                "text": "resultado",
                "speaker": "Participante B",
                "speaker_index": 1,
                "speaker_turn_index": 0,
                "diarization_reason": "model_overlap",
                "diarization_confidence": 1.0,
            }]
            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch("src.transcriptor.transcribe_one_chunk", return_value=[{"start_local": 0.0, "end_local": 1.0, "text": "resultado"}]), \
                    patch("src.transcriptor.require_sherpa_diarization", return_value="listo"), \
                    patch("src.transcriptor.create_sherpa_diarizer", return_value=object()), \
                    patch("src.transcriptor.normalize_to_wav_16k_mono"), \
                    patch("src.transcriptor.diarize_sherpa", return_value=model_result) as diarize_model:
                run_pipeline(args)

            output = json.loads((root / "salida" / "audio_transcripcion_final.json").read_text(encoding="utf-8"))
            self.assertEqual(output["meta"]["diarization_backend"], "sherpa")
            self.assertEqual(output["segments"][0]["speaker"], "Participante B")
            diarize_model.assert_called_once()

    def test_missing_diarization_model_fails_before_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)
            args.diarize = True

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch(
                        "src.transcriptor.require_sherpa_diarization",
                        side_effect=RuntimeError("ejecuta --setup_diarization_models"),
                    ), \
                    patch("src.transcriptor.load_whisper_model") as load_model:
                with self.assertRaisesRegex(RuntimeError, "setup_diarization_models"):
                    run_pipeline(args)
            load_model.assert_not_called()

    def test_invalid_sherpa_runtime_fails_before_whisper_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)
            args.diarize = True

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.require_sherpa_diarization", return_value="archivos presentes"), \
                    patch(
                        "src.transcriptor.create_sherpa_diarizer",
                        side_effect=RuntimeError("modelo ONNX corrupto"),
                    ), \
                    patch("src.transcriptor.load_whisper_model") as load_model:
                with self.assertRaisesRegex(RuntimeError, "ONNX corrupto"):
                    run_pipeline(args)
            load_model.assert_not_called()

    def test_completed_resume_does_not_reload_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()) as load_model, \
                    patch("src.transcriptor.transcribe_one_chunk", return_value=[{"start_local": 0.0, "end_local": 1.0, "text": "resultado"}]) as transcribe:
                run_pipeline(args)
                run_pipeline(args)

            self.assertEqual(load_model.call_count, 1)
            self.assertEqual(transcribe.call_count, 1)

    def test_resume_discards_uncommitted_partial_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch(
                        "src.transcriptor.transcribe_one_chunk",
                        side_effect=[
                            [{"start_local": 0.0, "end_local": 1.0, "text": "antes del corte"}],
                            [{"start_local": 0.0, "end_local": 1.0, "text": "reintento limpio"}],
                        ],
                    ):
                run_pipeline(args)
                state_path = root / "work" / "audio_estado.json"
                state = load_state(str(state_path))
                state["completed_chunks"] = []
                save_state(str(state_path), state)
                run_pipeline(args)

            partials = root / "work" / "audio_partials.jsonl"
            lines = [json.loads(line) for line in partials.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["text"] for line in lines], ["reintento limpio"])

    def test_corrupt_confirmed_partial_retranscribes_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split), \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch(
                        "src.transcriptor.transcribe_one_chunk",
                        side_effect=[
                            [{"start_local": 0.0, "end_local": 1.0, "text": "resultado inicial"}],
                            [{"start_local": 0.0, "end_local": 1.0, "text": "resultado recuperado"}],
                        ],
                    ) as transcribe:
                run_pipeline(args)
                partials = root / "work" / "audio_partials.jsonl"
                partials.write_text("{truncado", encoding="utf-8")
                run_pipeline(args)

            output = json.loads((root / "salida" / "audio_transcripcion_final.json").read_text(encoding="utf-8"))
            self.assertEqual([item["text"] for item in output["segments"]], ["resultado recuperado"])
            self.assertEqual(transcribe.call_count, 2)

    def test_force_reuse_regenerates_when_listed_chunk_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            args = pipeline_args(audio, root)

            with patch("src.transcriptor.ensure_ffmpeg"), \
                    patch("src.transcriptor.probe_audio_info", return_value={"sample_rate": 16000, "channels": 1, "duration_s": 1.0}), \
                    patch("src.transcriptor.split_audio_fixed", side_effect=self._fake_split) as split_audio, \
                    patch("src.transcriptor.load_whisper_model", return_value=object()), \
                    patch("src.transcriptor.transcribe_one_chunk", return_value=[{"start_local": 0.0, "end_local": 1.0, "text": "resultado"}]):
                run_pipeline(args)
                chunk = root / "work" / "audio_chunks" / "audio_chunk_0000.wav"
                chunk.rename(chunk.with_name("audio_chunk_9999.wav"))
                args.chunk_s = 60
                args.force_reuse_chunks = True
                run_pipeline(args)

            self.assertEqual(split_audio.call_count, 2)


class DiarizationUtilsTests(unittest.TestCase):
    def test_setup_extracts_root_model_and_validates_both_downloads(self) -> None:
        payload = b"\0" * 1_000_000

        def fake_download(url, destination, printer):
            if destination.name.endswith(".tar.bz2"):
                with tarfile.open(destination, "w:bz2") as bundle:
                    member = tarfile.TarInfo("model.int8.onnx")
                    member.size = len(payload)
                    bundle.addfile(member, io.BytesIO(payload))
            else:
                destination.write_bytes(payload)

        with tempfile.TemporaryDirectory() as tmpdir, \
                patch("src.diarization_utils._download_file", side_effect=fake_download):
            installed = setup_sherpa_diarization_models(tmpdir, lambda _: None)

            self.assertEqual(set(installed), {"segmentation", "embedding"})
            self.assertTrue(all(Path(path).stat().st_size >= 1_000_000 for path in installed.values()))

    def test_sherpa_constructor_error_is_reported_as_invalid_models(self) -> None:
        config = MagicMock()
        config.validate.return_value = True
        fake_module = SimpleNamespace(
            OfflineSpeakerSegmentationPyannoteModelConfig=MagicMock(return_value=object()),
            OfflineSpeakerSegmentationModelConfig=MagicMock(return_value=object()),
            SpeakerEmbeddingExtractorConfig=MagicMock(return_value=object()),
            FastClusteringConfig=MagicMock(return_value=object()),
            OfflineSpeakerDiarizationConfig=MagicMock(return_value=config),
            OfflineSpeakerDiarization=MagicMock(side_effect=OSError("invalid protobuf")),
        )
        with patch("src.diarization_utils.sherpa_diarization_ready", return_value=(True, "listo")), \
                patch("src.diarization_utils.importlib.import_module", return_value=fake_module):
            with self.assertRaisesRegex(RuntimeError, "cargar los modelos"):
                create_sherpa_diarizer("models", 2)

    def test_missing_sherpa_setup_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
                patch("src.diarization_utils.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "setup_diarization_models"):
                require_sherpa_diarization(tmpdir)

    def test_complete_sherpa_setup_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
                patch("src.diarization_utils.importlib.util.find_spec", return_value=object()):
            for model_path in sherpa_model_paths(tmpdir).values():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with model_path.open("wb") as model_file:
                    model_file.seek(1_000_000 - 1)
                    model_file.write(b"\0")
            self.assertIn("disponible", require_sherpa_diarization(tmpdir))

    def test_truncated_sherpa_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
                patch("src.diarization_utils.importlib.util.find_spec", return_value=object()):
            for model_path in sherpa_model_paths(tmpdir).values():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"incompleto")
            with self.assertRaisesRegex(RuntimeError, "incompletos"):
                require_sherpa_diarization(tmpdir)

    def test_model_turns_are_aligned_by_maximum_overlap(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "text": "hola"},
            {"start": 2.0, "end": 4.0, "text": "respuesta"},
        ]
        turns = [
            {"start": 0.0, "end": 2.2, "speaker_index": 4},
            {"start": 2.2, "end": 4.0, "speaker_index": 9},
        ]
        diarized = assign_speakers_from_turns(segments, turns)
        self.assertEqual([item["speaker"] for item in diarized], ["Participante A", "Participante B"])
        self.assertEqual([item["speaker_turn_index"] for item in diarized], [0, 1])
        self.assertTrue(all(item["diarization_reason"] == "model_overlap" for item in diarized))

    def test_segment_without_model_overlap_remains_unassigned(self) -> None:
        segments = [{"start": 10.0, "end": 11.0, "text": "sin turno detectado"}]
        turns = [{"start": 0.0, "end": 1.0, "speaker_index": 0}]
        diarized = assign_speakers_from_turns(segments, turns)
        self.assertIsNone(diarized[0]["speaker"])
        self.assertIsNone(diarized[0]["speaker_index"])
        self.assertEqual(diarized[0]["diarization_reason"], "model_no_overlap")
        self.assertEqual(diarized[0]["diarization_confidence"], 0.0)

    def test_overlapping_transcript_segments_keep_all_candidate_turns(self) -> None:
        segments = [
            {"start": 0.0, "end": 3.0, "text": "primero"},
            {"start": 2.0, "end": 4.0, "text": "segundo"},
        ]
        turns = [
            {"start": 0.0, "end": 2.5, "speaker_index": 0},
            {"start": 2.5, "end": 4.0, "speaker_index": 1},
        ]
        diarized = assign_speakers_from_turns(segments, turns)
        self.assertEqual(
            [item["speaker"] for item in diarized],
            ["Participante A", "Participante B"],
        )

    def test_speaker_label_supports_letters_and_overflow(self) -> None:
        self.assertEqual(speaker_label(0), "Participante A")
        self.assertEqual(speaker_label(26), "Participante S27")

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
