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

Catálogos: `data/raw/SRI_Vehiculos_DD.xlsx` (diccionario oficial del SRI, se coloca a mano en `data/raw/`)
aporta cantones, provincias y colores. El cantón viene como código en 2021-2025 y como nombre en 2026;
se unifica en `canton_codigo`, `canton_nombre` y `provincia`. Salidas extra: `dim_canton.parquet`,
`dim_color.parquet` y `data_dictionary.json` (definiciones de variables, útil para indexar en el RAG).
Casos sin resolver: nombres ambiguos de 2026 (BOLIVAR, OLMEDO), CHAGUARPAMBA y el código 22302, que
no está en el catálogo (su provincia sí se deduce del prefijo).

## Etapa 3: capa analítica y corpus

```
python scripts/build_corpus.py                       # prueba: 2025-2026, AUTOMOVIL/CAMION/CAMIONETA
python scripts/build_corpus.py --years 2021 2022 2023 2024 2025 2026 --clases AUTOMOVIL CAMION CAMIONETA
```

- `src/analytics/db.py`: vista DuckDB `vehiculos` sobre el Parquet (año y mes salen de `fecha_proceso`).
- `src/analytics/documents.py`: documentos de texto con metadatos (mercado mensual, marca anual/mensual,
  provincia, combustible, comparativo del mismo periodo y diccionario) en `data/processed/corpus/`.
  Las cifras las calcula SQL; los años incompletos se marcan y se comparan por el mismo rango de meses.
- `src/analytics/query.py`: ejecutor SQL de solo lectura para respuestas numéricas exactas.

### Boletines AEADE (PDF)

Los boletines mensuales de AEADE (`data/raw/Boletin-de-Prensa-<Mes>-2026.pdf`, exportados de Power BI)
se agregan al corpus con `src/ingestion/aeade_pdf.py` (un documento por página, con `fuente` y `pagina`).
Se corrige que las tablas traigan el encabezado del mes anterior, se reconstruyen las series mensuales de
los gráficos de barras y se advierte que los gráficos circulares pierden la asociación etiqueta-porcentaje.
Las cifras de AEADE y las del SRI procesado no coinciden (distinta cobertura y definición de venta):
el RAG debe citar la fuente y no mezclarlas.
