# whisper-local-transcriber

Transcripción local robusta en CPU para audios largos, con diarización ligera.
Diseñado para flujos de trabajo **offline**, priorizando la privacidad y el
control total de la información.

## 🚀 Características

- Transcripción local utilizando OpenAI Whisper
- Ejecución en CPU (no requiere GPU)
- Diarización ligera de hablantes
- Modo asistente interactivo
- Espacio de trabajo limpio y salidas organizadas
- Licencia GPLv3

## 📦 Instalación

Clona el repositorio e instala las dependencias:

```
git clone https://github.com/Januka19/whisper-local-transcriber.git
cd whisper-local-transcriber
pip install -r requirements.txt
```

## ▶️ Uso

Ejecuta el script principal:

```
bash run.sh
```
Sigue las instrucciones del asistente para transcribir archivos de audio de forma local

## 📁 Estructura del proyecto

whisper-local-transcriber/
├── src/                     # Lógica principal de la aplicación
├── .github/                 # Estándares de comunidad y contribución
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── SECURITY.md
│   └── ISSUE_TEMPLATE/
├── README.md
├── LICENSE
├── requirements.txt
└── run.sh

## 🤝 Contribuciones

Las contribuciones son bienvenidas y valoradas.
Antes de contribuir, por favor revisa:

📘 Código de Conducta
🛠️ Guía de Contribución
🔐 Política de Seguridad

Toda la documentación relacionada con la comunidad se encuentra centralizada en
la carpeta .github/.

## 🔐 Seguridad

Si identificas una vulnerabilidad de seguridad, por favor repórtala de manera
responsable.
Consulta la Política de Seguridad para más detalles.

## 📄 Licencia

Este proyecto se distribuye bajo la licencia GNU General Public License v3.0.
Consulta el archivo LICENSE para más información.

## 🧭 Hoja de ruta (corto plazo)

Mejorar la precisión de la diarización
Exportación opcional a formatos JSON y SRT
Cobertura básica de pruebas
Mejora continua de la documentación

## 📌 Versionado

Versión actual: v0.2.1

Esta versión se enfoca en la estandarización del proyecto, mejoras de
documentación y preparación para la colaboración con la comunidad.
No se incluyen cambios funcionales.
