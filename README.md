# whisper-local-transcriber

🎙️ **Transcriptor local robusto en CPU para audios largos**, con **diarización ligera por turnos**.  
Diseñado para entrevistas, reuniones y trabajo de campo, **sin depender de la nube ni GPU**.

---

## 🎯 Objetivo
Proveer un sistema **estable, mantenible y 100% local** para transcribir audios largos en español (y otros idiomas), optimizado para laptops comunes, con salidas listas para análisis posterior.

---

## ✨ Características
- Transcripción **100% local** (CPU-only)
- Optimizado para **audios largos**
- Reanudación automática si el proceso se interrumpe
- **Diarización simple por turnos** (Participante A / B / C)
- Salidas en **TXT** y **JSON**
- Modo **asistido por consola**
- Licencia **GPLv3 (copyleft)**

> ⚠️ **Nota**  
> La diarización es **ligera**, basada en pausas y duración.  
> No realiza identificación acústica de voces.

---

## 🖥️ Requisitos

### Sistema
- Linux (probado en Ubuntu)
- Python 3.9+
- `ffmpeg` y `ffprobe`
### Instalación en Ubuntu
```
sudo apt update
sudo apt install -y ffmpeg
```
---

## 🚀 Quick Start (Ubuntu)

### 1️⃣ Clonar el repositorio
```
git clone https://github.com/Januka19/whisper-local-transcriber.git
cd whisper-local-transcriber
```
### 2️⃣ Crear entorno virtual
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3️⃣ Ejecutar el transcriptor
```
./run.sh
```
El script se ejecuta en modo asistido y solicitará:
Ruta del archivo de audio
Idioma
Número de participantes
Parámetros recomendados para ejecución en CPU
Los resultados se guardan automáticamente en la carpeta salida/.

---

## 📁 Estructura del proyecto
```text
whisper-local-transcriber/
├── src/
│   └── transcriptor.py      # Núcleo del sistema de transcripción
├── run.sh                   # Punto de entrada único
├── requirements.txt         # Dependencias Python
├── README.md
├── LICENSE
├── work/                    # Archivos temporales (no versionado)
├── salida/                  # Resultados finales
└── logs/                    # Logs de ejecución
```
---

## 📄 Salidas
Por cada audio procesado, el sistema genera los siguientes archivos en la carpeta `salida/`:

- `*_transcripcion_final.txt`  
  Transcripción completa en texto plano, con marcas de turnos (Participante A/B/C).

- `*_transcripcion_final.json`  
  Transcripción estructurada en formato JSON, útil para análisis posterior,
  procesamiento con IA generativa o integración con otros sistemas.

---

## 🔒 Licencia
Este proyecto se distribuye bajo **GNU GPL v3**.

Cualquier modificación o redistribución debe mantenerse bajo la misma licencia  
y publicar el código fuente correspondiente.

---

## 🤝 Contribuciones
Las contribuciones son bienvenidas mediante **issues** o **pull requests**.

Puedes proponer:
- mejoras en la diarización por turnos
- optimizaciones de rendimiento en CPU
- nuevos formatos de salida
- mejoras de usabilidad y documentación

---

## 🧭 Roadmap
- Mejora de la diarización por turnos
- Modo no interactivo (`--audio archivo.wav`)
- Exportación a Markdown / DOCX
- Optimización adicional para ejecución en CPU
- Mejora de mensajes y validaciones para personas usuarias no técnicas

---

## 📌 Estado del proyecto
🟢 **Estable y probado en uso real**  
🟡 **En mejora continua**

