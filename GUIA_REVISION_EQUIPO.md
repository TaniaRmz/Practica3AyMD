# Guía de revisión para el equipo

Este paquete es un borrador de trabajo. El PDF final no se ha generado para permitir
que el equipo revise y apruebe las decisiones antes de cerrar la entrega.

## Orden sugerido de revisión

1. Leer `reports/reporte_practica3.md`.
2. Revisar `notebooks/01_eda.ipynb` y sus gráficas.
3. Revisar `notebooks/02_medidas_descriptivas.ipynb` y sus tablas.
4. Consultar los scripts de `src/cleaning/` para validar las reglas de limpieza.
5. Consultar `src/analysis/medidas_descriptivas.py` para revisar las fórmulas.

## Decisiones que requieren aprobación del equipo

- Eliminación de 4,374 duplicados crudos y 38 duplicados posteriores a la
  normalización.
- Conversión de edades fuera de 9-97 y códigos 98/99 a valores nulos.
- Decisión de no imputar `edad_primer_union` debido a 90.6% de datos faltantes.
- Imputación de `num_hijos` faltante con cero, manteniendo una bandera de auditoría.
- Tratamiento de `ingreso_pareja`: ceros estructurales, medianas estatales y
  conservación del valor censurado 999997.
- Clasificación de las siete variables principales.
- Interpretación de las diferencias entre resultados simples y ponderados.
- Redacción de las respuestas éticas y del cuestionario.

## Archivos no incluidos en el repositorio

El CSV de `data/data-raw/` se comparte por separado y no debe subirse a GitHub. El
entorno `.venv` tampoco se comparte: debe recrearse con `requirements.txt`.

## Cómo reproducir el análisis

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.cleaning.diagnostico
python -m src.cleaning.limpiar_datos
python -m src.visualization.eda
python -m src.analysis.medidas_descriptivas
```

Los notebooks ya contienen salidas ejecutadas para facilitar la revisión sin instalar
dependencias.
