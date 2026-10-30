# Evaluación del asistente

Generado el 2026-10-29 con `python scripts/make_report.py` a partir de los resultados medidos
(`data/processed/*.json`). Corpus: SRI 2025-2026 (automóvil, camión y camioneta) y boletines AEADE de julio a
septiembre de 2026. Todo corre en local: BGE-M3 para embeddings y modelos de Ollama para generación.

## 1. Recuperación

50 consultas. `hit@1`: el documento correcto es el primero; `hit@5`: está entre los 5 primeros; `mrr`: media del inverso de la posición.

| Configuración | hit@1 | hit@5 | MRR |
|---|---|---|---|
| bm25 | 0.52 | 0.58 | 0.555 |
| dense | 0.68 | 0.86 | 0.755 |
| dense+meta | 0.76 | 0.88 | 0.810 |
| hybrid_rrf | 0.52 | 0.66 | 0.585 |
| hybrid+meta | 0.82 | 0.88 | 0.848 |
| hybrid+meta bm25=.5 | 0.82 | 0.90 | 0.846 |
| hybrid+meta+rerank | 0.82 | 0.88 | 0.844 |

**hit@5 por categoría**

| Configuración | marca_anual (6) | mensual (6) | provincia (4) | combustible (4) | ranking (4) | comparativo (3) | diccionario (4) | aeade (10) | termino_exacto (9) |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.00 | 0.83 | 0.25 | 0.00 | 0.00 | 0.67 | 1.00 | 0.80 | 1.00 |
| dense | 0.83 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.44 |
| dense+meta | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.44 |
| hybrid_rrf | 0.00 | 0.83 | 1.00 | 0.00 | 0.25 | 1.00 | 1.00 | 1.00 | 0.67 |
| hybrid+meta | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.56 |
| hybrid+meta bm25=.5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.56 |
| hybrid+meta+rerank | 0.67 | 0.83 | 1.00 | 0.75 | 1.00 | 0.67 | 1.00 | 1.00 | 0.89 |

## 2. Text-to-SQL (cifras exactas)

Exactitud de ejecución: se compara el resultado de la consulta que escribe el modelo contra el de una consulta de
referencia escrita a mano. Un SQL distinto pero con el mismo resultado cuenta como correcto. La comparación tolera
columnas adicionales y diferencias de redondeo (por ejemplo `21880,3` frente a `21880`), pero no diferencias en los
valores; también se informa la comparación estricta.

**qwen2.5:3b-instruct**: 26/31 correctas (84 %) (comparación estricta, que exige columnas y redondeo idénticos: 24/31); 3.5 s por consulta en promedio. Por categoría: conteo 10/10, filtros_multiples 5/5, tiempo 3/3, porcentaje 2/4, ranking 3/5, estadistica 3/4.

- Falla: *¿Qué porcentaje de los registros de 2026 fueron eléctricos?* — esperado `[[5.0]]`, obtenido `[[100.0]]`
- Falla: *¿Cuánto creció, en porcentaje, el total de registros de enero a septiembre de 2026 frente al mismo periodo de 2025?* — esperado `[[40.9]]`, obtenido `[[7821899.0]]`
- Falla: *¿Cuáles son las 5 provincias con más registros en 2025?* — esperado `[['PICHINCHA', 34390.0], ['GUAYAS', 21843.0], ['AZUAY', 7283.0], ['TUNGURAHUA', 5169.0], ['MANABI', 2393.0]]`, obtenido `[['QUITO', 2025.0, 22537.0], ['GUAYAQUIL', 2025.0, 18248.0], ['AMBATO', 2025.0, 4964.0], ['SAN MIGUEL DE LOS BANCOS', 2025.0, 3788.0], ['PEDRO VICENTE MALDONADO', 2025.0, 3285.0]]`
- Falla: *¿Cuál es el combustible con más registros en 2026?* — esperado `[['DIESEL']]`, obtenido `[['GASOLINA', 20.0]]`
- Falla: *¿Cuál es el avalúo mediano de los Kia en 2026?* — esperado `[[16199.0]]`, obtenido `[[875038.1]]`

**qwen2.5:7b-instruct**: 31/31 correctas (100 %) (comparación estricta, que exige columnas y redondeo idénticos: 26/31); 17.5 s por consulta en promedio. Por categoría: conteo 10/10, filtros_multiples 5/5, tiempo 3/3, porcentaje 4/4, ranking 5/5, estadistica 4/4.

## 3. Asistente completo

Preguntas de extremo a extremo (router, SQL o documentos, guardrail). Una respuesta es correcta si contiene las
cifras y textos esperados, no contiene valores prohibidos y quedó verificada; las preguntas fuera de alcance deben
rechazarse.

| Modelo | Correctas | sql | documentos | aeade | rechazo | Ruta esperada | s por pregunta |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b-instruct | 20/24 (83 %) | 8/10 | 4/4 | 4/5 | 4/5 | 23/24 | 9.1 |
| qwen2.5:7b-instruct | 22/24 (92 %) | 9/10 | 4/4 | 4/5 | 5/5 | 24/24 | 87.3 |

**Fallos de qwen2.5:3b-instruct**

- *¿Qué porcentaje de los registros de 2026 fueron eléctricos?* (ruta sql): falta la cifra 5
- *¿Cuánto creció, en porcentaje, el total de registros de enero a septiembre de 2026 frente al mismo periodo de 2025?* (ruta sql): falta la cifra 40.9
- *¿Cuánto crecieron las ventas de híbridos entre enero y septiembre según AEADE?* (ruta rag): falta la cifra 81
- *¿Cuántos vehículos usados se vendieron en 2026?* (ruta sql): debia rechazar y respondio

**Fallos de qwen2.5:7b-instruct**

- *¿Cuánto creció Chevrolet entre enero y septiembre de 2026 frente a 2025?* (ruta sql): falta la cifra 9.6
- *¿Cuántos vehículos se vendieron en septiembre de 2026 según el boletín de AEADE?* (ruta rag): falta la cifra 15649

## 4. Hallazgos y límites

- **Recuperación.** La búsqueda híbrida solo supera a los embeddings cuando se le suma el entendimiento de la consulta
  (filtros por año, marca, provincia y combustible). Sin él, BM25 arrastra la fusión hacia los resúmenes mensuales.
  El reranker no justifica sus 2 GB: ayuda con términos exactos y daña las consultas estructuradas. Un peso adaptativo de
  BM25 para términos raros se probó y no funcionó (RRF premia el consenso), por lo que se retiró.
- **Text-to-SQL.** El modelo de 7B resuelve las 31 preguntas; el de 3B escribe SQL erróneo con seguridad
  (por ejemplo, un porcentaje de 100 en lugar de 5). Para cifras exactas se recomienda el 7B. La interfaz muestra siempre la
  consulta para poder auditarla.
- **Guardrail.** Verifica que cada cifra exista en el fragmento citado, no que ese fragmento sea el adecuado para la
  pregunta. Es la causa de los fallos que quedan en las preguntas sobre AEADE.
- **Preguntas ambiguas.** "¿Cuánto creció Chevrolet?" admite dos lecturas (unidades o porcentaje). No se ajustó el criterio
  para que pase.
- **Sesgo de la medición.** Las reglas del parser y del prompt se afinaron viendo los fallos de estos mismos sets; un
  conjunto de validación separado daría una cifra más conservadora.
- **Alcance.** SRI 2025-2026 (automóvil, camión y camioneta) y tres boletines de AEADE. Las cifras de ambas fuentes no
  coinciden y no se concilian.

