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
