# AGENTS.md

Contexto para IAs y agentes de código que trabajen en este repositorio.

## Proyecto

`whisper-local-transcriber` es un transcriptor local para audios largos. El foco
principal es ejecutar en CPU, funcionar offline cuando los modelos ya están
disponibles, preservar privacidad y soportar reanudación de trabajos largos.

La lógica principal vive en `src/transcriptor.py`. El script `run.sh` prepara el
entorno, valida dependencias del sistema y ejecuta el transcriptor. La
interfaz puede usarse como asistente interactivo o como CLI reproducible.

## Principios de diseño

- Mantener el flujo CPU-friendly; no asumir GPU ni servicios externos.
- Evitar cambios que descarguen modelos, suban audios o dependan de red durante
  una transcripción normal.
- Conservar `resume`: no mezclar parciales generados con configuraciones
  incompatibles.
- Preferir cambios pequeños y verificables; este proyecto suele usarse con
  audios largos, así que los errores aparecen tarde y cuestan tiempo.
- Mantener `faster_whisper` como dependencia de ejecución, no como requisito para
  importar utilidades, pedir `--help` o correr pruebas unitarias livianas.

## Arquitectura rápida

- `src/transcriptor.py`: pipeline completo.
  - Resolución de ruta de audio.
  - Validación de `ffmpeg`/`ffprobe`.
  - Normalización opcional a WAV 16 kHz mono.
  - División en chunks con overlap.
  - Estado de reanudación y parciales JSONL.
  - Carga perezosa de modelo principal y fallback.
  - Transcripción con `faster-whisper`.
  - Deduplicación, postproceso y diarización local con Sherpa ONNX.
  - Escritura de salidas TXT/JSON.
- `run.sh`: runner Bash robusto para crear/usar `.venv`, instalar requirements y
  llamar a `src/transcriptor.py`.
- `tests/`: pruebas unitarias pequeñas, pensadas para no requerir modelos ni
  descargas.
- `work/`, `salida/`, `logs/`: carpetas de ejecución generadas localmente; no
  deben versionarse.

## Dependencias importantes

Runtime:

```bash
pip install -r requirements.txt
```

Desarrollo:

```bash
pip install -r requirements-dev.txt
```

Sistema:

- `ffmpeg`
- `ffprobe`

## Comandos útiles

Ver ayuda del CLI sin cargar modelos:

```bash
python src/transcriptor.py --help
```

Ejecutar modo asistente:

```bash
./run.sh
```

Ejecutar un audio por CLI:

```bash
./run.sh ruta/al/audio.mp3 --language es
```

Pruebas:

```bash
pytest
```

Validaciones rápidas cuando `pytest` no esté instalado:

```bash
python3 -m py_compile src/transcriptor.py tests/test_transcriptor.py
python3 -c 'import src.transcriptor; print("import-ok")'
python3 src/transcriptor.py --help
```

## Convenciones de implementación

- No importes `faster_whisper` al nivel superior. Usa el helper de carga perezosa
  para que `--help` y las pruebas sigan funcionando sin la dependencia pesada.
- Si agregas parámetros que cambian la transcripción, añádelos a la firma de
  configuración de `resume` (`config_signature`) y a la metadata final.
- Si agregas opciones CLI, considera también el asistente interactivo y el
  README.
- Mantén valores por defecto conservadores. El comportamiento actual debe seguir
  funcionando sin flags nuevos.
- No rompas la compatibilidad con rutas locales de modelos o IDs de Hugging Face.
- Evita procesamiento global costoso al importar el módulo.
- Si una optimización toca audio/chunks, valida cuidadosamente el reuso de
  metadata y estado.

## Salidas y estado

El pipeline genera archivos en:

```text
work/     estado, chunks, WAV normalizado, parciales JSONL
salida/   TXT y JSON finales
logs/     logs de ejecución
```

`--clean` elimina intermedios al final. No agregues estas carpetas al repo salvo
que el usuario lo pida explícitamente.

## Rendimiento

Opciones relevantes:

- `--compute_type int8`: recomendado en CPU por defecto.
- `--cpu_threads 0`: deja que `faster-whisper` decida automáticamente.
- `--num_workers 1`: conserva el comportamiento estable por defecto.
- `--chunk_s` y `--overlap_s`: afectan calidad, velocidad y reuso de chunks.
- `--vad_filter`: puede reducir trabajo en silencios, pero cambia segmentos.

La carga del fallback debe permanecer perezosa: solo cargarlo si falla el modelo
principal.

## Pruebas y cautelas

- Las pruebas no deben requerir descargar modelos ni procesar audios grandes.
- Para funciones auxiliares, usa fixtures pequeñas y datos sintéticos.
- Antes de commitear, revisa:

```bash
git diff --check
python3 -m py_compile src/transcriptor.py tests/test_transcriptor.py
python3 src/transcriptor.py --help
```

- Si `pytest` está disponible, ejecútalo completo.
- Si no está disponible, dilo claramente en el resumen final.

## Git

- Revisa `git status --short --branch` antes de editar y antes de commitear.
- No reviertas cambios existentes del usuario.
- Haz stage explícito de archivos relacionados; evita `git add -A` si el árbol
  está mezclado.
- El repositorio puede estar adelantado respecto a `origin/main` si el push está
  bloqueado por credenciales locales.
