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

## Etapa 2: ETL a Parquet

```
python scripts/run_etl.py            # data/raw/*.csv -> data/processed/vehiculos_nuevos/
```

Unifica el esquema de los 6 años (encabezados, codificación UTF-8/latin-1, fechas, decimales con coma,
caracteres dañados en el CSV 2025) y escribe Parquet particionado por `anio_archivo` más
`data/processed/etl_report.json` con la calidad por año (≈3.2 M filas, ~18 MB).

Limitación conocida: el cantón viene como código SRI en 2021-2025 y como nombre en 2026
(`canton_codigo` / `canton_nombre`); no hay tabla de equivalencia.
