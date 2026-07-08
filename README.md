# whisper-local-transcriber

Transcriptor local para audios largos basado en `faster-whisper`, pensado para
ejecutarse en CPU y trabajar offline. Incluye asistente interactivo, reanudacion
de trabajos, normalizacion de audio, postproceso de texto y diarizacion ligera
por reglas.

## Caracteristicas

- Transcripcion local con `faster-whisper`.
- Optimizado para CPU con modelo turbo INT8 por defecto.
- Asistente interactivo si se ejecuta sin argumentos.
- CLI reproducible para automatizar transcripciones.
- Normalizacion inteligente a WAV 16 kHz mono solo cuando el audio lo necesita.
- Division en chunks con overlap, estado de reanudacion y parciales JSONL.
- Postproceso opcional: limpieza de muletillas, reemplazos y fusion de segmentos.
- Diarizacion ligera por turnos y pausas, con motivo y confianza por segmento en JSON.
- Salidas organizadas en texto y JSON.
- Pruebas automatizadas con `pytest`.

## Requisitos

- Python 3.11 o 3.12 recomendado.
- `ffmpeg` y `ffprobe` disponibles en `PATH`.
- Linux/macOS o un entorno compatible con Bash para usar `run.sh`.

Instala `ffmpeg` si hace falta:

```bash
# Fedora/RHEL
sudo dnf install -y ffmpeg-free ffmpeg-free-devel

# Ubuntu/Debian
sudo apt-get install -y ffmpeg
```

Tambien puedes pedirle al runner que intente instalar dependencias del sistema:

```bash
./run.sh --system-deps
```

## Instalacion

```bash
git clone https://github.com/Januka19/whisper-local-transcriber.git
cd whisper-local-transcriber
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El script `run.sh` puede crear y mantener la venv automaticamente. El primer
arranque instala dependencias; los siguientes omiten `pip install` si
`requirements.txt` no cambio y las dependencias siguen disponibles:

```bash
./run.sh
```

Para reinstalar dependencias o recrear la venv desde cero:

```bash
./run.sh --force-install
./run.sh --rebuild-venv
```

## Uso

Modo asistente:

```bash
./run.sh
```

Modo CLI:

```bash
./run.sh ruta/al/audio.mp3
```

Ejemplo con opciones frecuentes:

```bash
./run.sh entrevista.wav \
  --model turbo-int8 \
  --language es \
  --chunk_s 45 \
  --overlap_s 0.4 \
  --diarize \
  --num_speakers 2
```

Tambien puedes ejecutar el modulo directamente:

```bash
python src/transcriptor.py ruta/al/audio.mp3 --language es
```

Si omites la ruta del audio, la aplicacion abre el asistente interactivo.

## Modelos

El modelo por defecto es `Zoont/faster-whisper-large-v3-turbo-int8-ct2`.
Tambien se aceptan alias cortos, IDs de Hugging Face o rutas locales.

Alias incluidos:

- `turbo-int8`
- `large-v3-turbo-int8`
- `large-v3-turbo-int8-ct2`
- `turbo`
- `large-v3-turbo`
- `large-v3`
- `large-v2`
- `medium`

El fallback por defecto es `medium`, util si el modelo principal falla o no cabe
en los recursos disponibles.

## Opciones utiles

```bash
--language es|en|auto
--model turbo-int8
--fallback_model medium
--compute_type int8
--cpu_threads 0
--num_workers 1
--chunk_s 45
--overlap_s 0.4
--beam 1
--normalize / --no-normalize
--resume / --no-resume
--vad_filter / --no-vad_filter
--diarize / --no-diarize
--num_speakers 2
--turn_gap_s 1.2
--force_turn_max_s 30.0
--review_diarization / --no-review_diarization
--clean
```

`--resume` esta activo por defecto. La configuracion se firma para evitar
mezclar parciales generados con parametros incompatibles.

`--cpu_threads 0` deja que `faster-whisper` decida automaticamente los hilos de
CPU. Puedes subirlo o fijarlo segun tu maquina; `--num_workers` controla workers
internos del modelo y por defecto conserva el comportamiento actual.

La diarizacion es local y ligera: alterna participantes cuando detecta pausas de
`--turn_gap_s` o cuando un turno supera `--force_turn_max_s`. Para evitar falsos
cambios, las interjecciones muy cortas despues de una pausa se conservan en el
turno actual con menor confianza. En JSON cada segmento diarizado incluye
`speaker`, `speaker_index`, `speaker_turn_index`, `diarization_reason` y
`diarization_confidence` para revisar asignaciones ambiguas.

La deteccion inicial de audio usa `ffprobe` para obtener formato y duracion en
una sola pasada. Al dividir en chunks, el runner evita crear un ultimo fragmento
que solo repite audio ya cubierto por el overlap, reduciendo trabajo innecesario
en CPU.

## Salidas

Por defecto se crean estas carpetas:

```text
work/     # estado, chunks, normalizados y parciales
salida/   # transcripcion final .txt y .json
logs/     # logs de ejecucion
```

Cada transcripcion produce:

- `<audio>_transcripcion_final.txt`
- `<audio>_transcripcion_final.json`
- logs con la configuracion usada y el progreso del pipeline.

Usa `--clean` si quieres eliminar intermedios al finalizar.

## Pruebas

Instala dependencias de desarrollo y ejecuta:

```bash
pip install -r requirements-dev.txt
pytest
```

## Estructura del proyecto

```text
whisper-local-transcriber/
├── src/
│   └── transcriptor.py
├── tests/
│   └── test_transcriptor.py
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
└── run.sh
```

## Licencia

Este proyecto se distribuye bajo la licencia GNU General Public License v3.0.
Consulta `LICENSE` para mas informacion.

## Version

Version actual: v0.3.0

Esta version documenta el flujo Release 3 e incorpora normalizacion inteligente,
alias de modelos adicionales, dependencias acotadas y cobertura basica de
pruebas.
