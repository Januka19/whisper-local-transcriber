# Whisper Local Transcriber (CPU)

Sistema local, robusto y mantenible para transcribir **audios largos** (entrevistas y reuniones)
en español, pensado para **laptops sin GPU**.  
Funciona 100% local, con **reanudación**, **salidas limpias** y **diarización ligera por turnos**
(tipo *Participante A / B*).

---

## 🎯 ¿Para qué sirve?
- Transcribir reuniones, entrevistas y misiones de campo
- Audios largos (horas), calidad media o baja
- Contextos profesionales, proyectos, consultoría y análisis
- Privacidad: **nada sale del equipo**

---

## ✨ Características principales
- Procesamiento **100% local (CPU)**
- Normalización opcional a WAV mono 16 kHz
- División en chunks con overlap (estable y robusto)
- Reanudación automática si el proceso se interrumpe
- Salidas en **TXT** y **JSON**
- Post-procesado opcional (limpieza básica)
- **Diarización simple por turnos** (Participante A/B/C…)
- Interfaz **modo asistido** por consola
- Ejecución en **un solo comando** (`./run.sh`)

> Nota: la diarización es ligera (basada en pausas y duración),
> no es diarización acústica por identificación de voz.

---

## 🖥️ Requisitos

### Sistema
- Linux (probado en Ubuntu)
- `ffmpeg` y `ffprobe`

Instalación en Ubuntu:
```bash
sudo apt update
sudo apt install -y ffmpeg

---
## 📄 Licencia
Este proyecto se distribuye bajo **GNU GPL v3**.

Cualquier modificación o redistribución debe mantenerse bajo la misma licencia
y publicar el código fuente correspondiente.

