# 3D Generator — Local & Free

Genera modelos 3D desde texto o imágenes y descárgalos para imprimir.

## Requisitos

- Windows 10/11
- Python 3.10 o 3.11
- Node.js 18+
- GPU NVIDIA con CUDA (recomendado: RTX 3070 Ti o superior, 8GB+ VRAM)

## Instalación

```bat
install.bat
```

La primera vez descarga los modelos de IA (~2 GB). Requiere internet.

## Uso

```bat
start.bat
```

Abre automáticamente `http://localhost:3000` en el navegador.

## Stack

| Capa | Tecnología |
|------|-----------|
| Frontend | Next.js 14 + React Three Fiber |
| Backend | Python FastAPI |
| Texto → 3D | Shap-E (OpenAI) |
| Imagen → 3D | TripoSR (Stability AI) |
| Exportación | STL, OBJ, GLB |

## Tiempos estimados (RTX 3070 Ti)

| Modo | Tiempo |
|------|--------|
| Imagen → 3D | ~2-5 segundos |
| Texto → 3D | ~30-60 segundos |
