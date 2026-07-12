# whisper-local-transcriber

Transcriptor local para audios largos basado en `faster-whisper`, pensado para
ejecutarse en CPU y trabajar offline. Incluye asistente interactivo, reanudacion
de trabajos, normalizacion de audio, postproceso de texto y diarizacion local
con Sherpa ONNX.

## Caracteristicas

- Transcripcion local con `faster-whisper`.
- Optimizado para CPU con modelo turbo INT8 por defecto.
- Asistente interactivo si se ejecuta sin argumentos.
- CLI reproducible para automatizar transcripciones.
- Normalizacion inteligente a WAV 16 kHz mono solo cuando el audio lo necesita.
- Division en chunks con overlap, estado de reanudacion y parciales JSONL.
- Postproceso opcional: limpieza de muletillas, reemplazos y fusion de segmentos.
- Diarizacion local con modelo Sherpa ONNX ligero.
- Salidas organizadas en texto y JSON.
- Pruebas automatizadas con `pytest`.

## Requisitos

- Python 3.9 como minimo; Python 3.11 o 3.12 recomendado.
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

El modelo por defecto es `large-v3-turbo`, el alias CTranslate2 soportado
directamente por `faster-whisper`. Se ejecuta con `compute_type=int8` para
mantener un buen equilibrio entre precision, velocidad y memoria en CPU.
Tambien se aceptan alias cortos, IDs de Hugging Face o rutas locales.

Alias incluidos:

- `turbo-int8`
- `large-v3-turbo-int8`
- `large-v3-turbo-int8-ct2`
- `turbo`
- `large-v3-turbo`
- `quality` / `max-quality` (modelo `large-v3` completo; mas preciso, pero
  considerablemente mas lento y pesado en CPU)
- `large-v3`
- `large-v2`
- `medium`
- `turbo-int8-legacy` (checkpoint anterior de Zoont, util si ya esta descargado
  y necesitas seguir completamente offline)

La primera ejecucion con el nuevo modelo por defecto puede descargarlo. Una vez
presente en la cache local, las transcripciones posteriores vuelven a funcionar
offline. Si solo tienes descargado el checkpoint usado por versiones anteriores,
usa temporalmente `--model turbo-int8-legacy`.

El fallback por defecto es `medium`, util si el modelo principal falla o no cabe
en los recursos disponibles.

## Diarizacion

La diarizacion usa exclusivamente el modelo local Sherpa ONNX y esta activa por
defecto. Nunca descarga pesos durante una transcripcion normal, por lo que debes
prepararlos una vez antes de usarla.

Prepara una vez los modelos de segmentacion Pyannote INT8 y embeddings TitaNet:

```bash
./run.sh --setup_diarization_models
```

Los pesos se guardan por defecto en `models/diarization/` y despues funcionan
offline. Si no necesitas diarizacion, puedes omitirla explicitamente:

```bash
./run.sh entrevista.wav --no-diarize
```

La aplicacion comprueba la dependencia y los pesos antes de iniciar la
transcripcion, evitando descubrir el problema despues de procesar un audio
largo. Puedes cambiar su ubicacion con `--diarization_model_dir`.

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
--diarization_model_dir models/diarization
--num_speakers 2
--review_diarization / --no-review_diarization
--clean
```

`--resume` esta activo por defecto. La configuracion se firma para evitar
mezclar parciales generados con parametros incompatibles.

`--cpu_threads 0` deja que `faster-whisper` decida automaticamente los hilos de
CPU. Puedes subirlo o fijarlo segun tu maquina; `--num_workers` controla workers
internos del modelo y por defecto conserva el comportamiento actual.

La diarizacion siempre es local. Sherpa detecta los turnos mediante modelos ONNX
y los alinea con los segmentos transcritos por mayor solapamiento temporal. En
JSON cada segmento diarizado incluye
`speaker`, `speaker_index`, `speaker_turn_index`, `diarization_reason` y
`diarization_confidence`; la metadata registra Sherpa como backend.
Si el modelo no produce ningun turno que solape un segmento, este queda sin
hablante en lugar de recibir una atribucion inventada.

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
