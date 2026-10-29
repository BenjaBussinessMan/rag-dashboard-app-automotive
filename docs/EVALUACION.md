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

**qwen2.5:3b-instruct**: 23/31 correctas (74 %); 3.7 s por consulta en promedio. Por categoría: conteo 10/10, filtros_multiples 4/5, tiempo 1/3, porcentaje 2/4, ranking 4/5, estadistica 2/4.

- Falla: *¿Cuántos Toyota Hilux se registraron en 2026?* — esperado `[[1824.0]]`, obtenido `[[0.0]]`
- Falla: *¿Cuántos vehículos se registraron entre enero y junio de 2026?* — esperado `[[50761.0]]`, obtenido `[[16679.0]]`
- Falla: *¿En qué mes de 2026 hubo más registros?* — esperado `[[4.0]]`, obtenido `[[4.0, 9589.0, 12.3]]`
- Falla: *¿Cuánto creció, en porcentaje, el total de registros de enero a septiembre de 2026 frente al mismo periodo de 2025?* — esperado `[[40.9]]`, obtenido `[[1674700.0]]`
- Falla: *¿Cuántos vehículos más se registraron en enero-septiembre de 2026 que en el mismo periodo de 2025?* — esperado `[[22707.0]]`, obtenido `[[55512.0, 78219.0]]`
- Falla: *¿Cuál es el combustible con más registros en 2026?* — esperado `[['DIESEL']]`, obtenido `[['GASOLINA', 20.0]]`
- Falla: *¿Cuál es el avalúo mediano de los Kia en 2026?* — esperado `[[16199.0]]`, obtenido `None`
- Falla: *¿Cuál es el avalúo promedio de los vehículos eléctricos en 2026?* — esperado `[[21880.0]]`, obtenido `[[21880.3]]`

**qwen2.5:7b-instruct**: 25/31 correctas (81 %); 9.8 s por consulta en promedio. Por categoría: conteo 10/10, filtros_multiples 4/5, tiempo 2/3, porcentaje 4/4, ranking 3/5, estadistica 2/4.

- Falla: *¿Cuántos Toyota Hilux se registraron en 2026?* — esperado `[[1824.0]]`, obtenido `[[0.0]]`
- Falla: *¿En qué mes de 2026 hubo más registros?* — esperado `[[4.0]]`, obtenido `[[4.0, 9589.0]]`
- Falla: *¿Cuál es la marca con más registros en Guayas en 2026?* — esperado `[['CHEVROLET']]`, obtenido `[['CHEVROLET', 3514.0]]`
- Falla: *¿Cuál es el combustible con más registros en 2026?* — esperado `[['DIESEL']]`, obtenido `[['DIESEL', 37264.0, 47.6]]`
- Falla: *¿Cuál es el avalúo promedio de los vehículos eléctricos en 2026?* — esperado `[[21880.0]]`, obtenido `[[21880.3]]`
- Falla: *¿Cuál es el cilindraje promedio de los vehículos a gasolina en 2025?* — esperado `[[1493.0]]`, obtenido `[[1493.4]]`

## 3. Asistente completo

Preguntas de extremo a extremo (router, SQL o documentos, guardrail). Una respuesta es correcta si contiene las
cifras y textos esperados, no contiene valores prohibidos y quedó verificada; las preguntas fuera de alcance deben
rechazarse.

| Modelo | Correctas | sql | documentos | aeade | rechazo | Ruta esperada | s por pregunta |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b-instruct | 17/24 (71 %) | 6/10 | 3/4 | 4/5 | 4/5 | 20/24 | 7.9 |
| qwen2.5:7b-instruct | 19/24 (79 %) | 8/10 | 3/4 | 4/5 | 4/5 | 21/24 | 47.9 |

**Fallos de qwen2.5:3b-instruct**

- *¿Cuáles son las 3 marcas con más registros en 2026?* (ruta rag): rechazo una pregunta contestable; falta la cifra 10977; falta la cifra 12198; falta la cifra 5433; falta el texto 'KIA'; falta el texto 'CHEVROLET'; falta el texto 'CHERY'
- *¿En qué provincia se registraron más vehículos BYD en 2026?* (ruta rag): falta la cifra 1013
- *¿Cuánto creció, en porcentaje, el total de registros de enero a septiembre de 2026 frente al mismo periodo de 2025?* (ruta sql): falta la cifra 40.9
- *¿En qué mes de 2026 hubo más registros?* (ruta rag): falta el texto 'abril'
- *¿Cuáles son las marcas más vendidas en 2026?* (ruta rag): rechazo una pregunta contestable; falta el texto 'KIA'; falta el texto 'CHEVROLET'
- *¿Cuánto crecieron las ventas de híbridos entre enero y septiembre según AEADE?* (ruta rag): falta la cifra 81
- *¿Cuántos vehículos usados se vendieron en 2026?* (ruta sql): debia rechazar y respondio

**Fallos de qwen2.5:7b-instruct**

- *¿Cuáles son las 3 marcas con más registros en 2026?* (ruta rag): rechazo una pregunta contestable; falta la cifra 10977; falta la cifra 12198; falta la cifra 5433; falta el texto 'KIA'; falta el texto 'CHEVROLET'; falta el texto 'CHERY'
- *¿En qué mes de 2026 hubo más registros?* (ruta rag): rechazo una pregunta contestable; falta el texto 'abril'
- *¿Cuáles son las marcas más vendidas en 2026?* (ruta rag): rechazo una pregunta contestable; falta el texto 'KIA'; falta el texto 'CHEVROLET'
- *¿Cuántos vehículos se vendieron en septiembre de 2026 según el boletín de AEADE?* (ruta rag): falta la cifra 15649
- *¿Cuántos vehículos usados se vendieron en 2026?* (ruta sql): debia rechazar y respondio


