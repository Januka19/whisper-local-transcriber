from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional


SHORT_GAP_CONFIDENCE = 0.95
GAP_CHANGE_CONFIDENCE = 0.85
FORCED_CHANGE_CONFIDENCE = 0.6
SHORT_SEGMENT_HOLD_CONFIDENCE = 0.7
SHORT_TURN_MAX_S = 0.8
SHORT_TURN_MAX_WORDS = 2


def speaker_label(i: int) -> str:
    if 0 <= i < 26:
        return f"Participante {chr(ord('A') + i)}"
    return f"Participante S{i+1}"


def _is_short_segment(segment: Dict[str, Any], start: float, end: float) -> bool:
    duration = max(0.0, end - start)
    words = len((segment.get("text") or "").split())
    return duration <= SHORT_TURN_MAX_S and words <= SHORT_TURN_MAX_WORDS


def _change_reason(
    segment: Dict[str, Any],
    start: float,
    end: float,
    gap: float,
    turn_len: float,
    turn_gap_s: float,
    force_turn_max_s: float,
) -> Optional[str]:
    if gap >= turn_gap_s:
        if _is_short_segment(segment, start, end):
            return "short_segment_hold"
        return "gap"
    if turn_len >= force_turn_max_s:
        return "max_turn"
    return None


def _confidence_for_reason(reason: str) -> float:
    if reason == "gap":
        return GAP_CHANGE_CONFIDENCE
    if reason == "max_turn":
        return FORCED_CHANGE_CONFIDENCE
    if reason == "short_segment_hold":
        return SHORT_SEGMENT_HOLD_CONFIDENCE
    return SHORT_GAP_CONFIDENCE


def diarize_light(segments: List[Dict[str, Any]], num_speakers: int, turn_gap_s: float, force_turn_max_s: float) -> List[Dict[str, Any]]:
    if not segments:
        return []
    num_speakers = max(1, int(num_speakers))
    turn_gap_s = max(0.0, float(turn_gap_s))
    force_turn_max_s = max(1.0, float(force_turn_max_s))

    out: List[Dict[str, Any]] = []
    current = 0
    turn_index = 0

    ordered = sorted(segments, key=lambda x: (float(x["start"]), float(x["end"])))
    turn_start = float(ordered[0]["start"])
    prev_end = float(ordered[0]["end"])

    first = dict(ordered[0])
    first["speaker"] = speaker_label(current)
    first["speaker_index"] = current
    first["speaker_turn_index"] = turn_index
    first["diarization_reason"] = "start"
    first["diarization_confidence"] = SHORT_GAP_CONFIDENCE
    out.append(first)

    for s in ordered[1:]:
        start = float(s["start"])
        end = float(s["end"])
        gap = max(0.0, start - prev_end)
        turn_len = max(0.0, prev_end - turn_start)

        reason = _change_reason(s, start, end, gap, turn_len, turn_gap_s, force_turn_max_s)
        if reason == "short_segment_hold":
            pass
        elif reason:
            turn_index += 1
            turn_start = start
            if num_speakers > 1:
                current = (current + 1) % num_speakers
        else:
            reason = "continuation"

        s2 = dict(s)
        s2["speaker"] = speaker_label(current)
        s2["speaker_index"] = current
        s2["speaker_turn_index"] = turn_index
        s2["diarization_reason"] = reason
        s2["diarization_confidence"] = _confidence_for_reason(reason)
        out.append(s2)
        prev_end = max(prev_end, end)

    return out


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
