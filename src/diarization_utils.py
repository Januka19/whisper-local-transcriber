from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List


def speaker_label(i: int) -> str:
    if 0 <= i < 26:
        return f"Participante {chr(ord('A') + i)}"
    return f"Participante S{i+1}"


def diarize_light(segments: List[Dict[str, Any]], num_speakers: int, turn_gap_s: float, force_turn_max_s: float) -> List[Dict[str, Any]]:
    if not segments:
        return []
    num_speakers = max(1, int(num_speakers))
    turn_gap_s = max(0.0, float(turn_gap_s))
    force_turn_max_s = max(1.0, float(force_turn_max_s))

    out: List[Dict[str, Any]] = []
    current = 0
    turn_start = float(segments[0]["start"])
    prev_end = float(segments[0]["end"])

    first = dict(segments[0])
    first["speaker"] = speaker_label(current)
    out.append(first)

    for s in segments[1:]:
        start = float(s["start"])
        end = float(s["end"])
        gap = start - prev_end
        turn_len = prev_end - turn_start

        change = (gap >= turn_gap_s) or (turn_len >= force_turn_max_s)
        if change:
            current = (current + 1) % num_speakers
            turn_start = start

        s2 = dict(s)
        s2["speaker"] = speaker_label(current)
        out.append(s2)
        prev_end = end

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
                i += 1
                continue

        printer("Comando no válido.\n")

    printer("Revisión finalizada.\n")
    return out
