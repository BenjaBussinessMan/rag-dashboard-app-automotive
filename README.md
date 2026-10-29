# Asistente Inteligente de Análisis del Mercado Automotriz

Proyecto RAG sobre datos abiertos del SRI Ecuador (vehículos nuevos).

## Etapa 1: ingesta

Descarga los CSV anuales (2021-2026) a `data/raw/` y genera `data/raw/manifest.json`
(filas, fecha máxima, sha256 por archivo).

```
python scripts/download_sri.py                  # todos los años
python scripts/download_sri.py --years 2025 2026
python scripts/download_sri.py --force          # re-descarga
```

Los años cerrados se omiten si el tamaño remoto coincide; el año en curso siempre se refresca.
Formato de los CSV: separador `;`, codificación `latin-1`.

Tests: `python -m pytest`
