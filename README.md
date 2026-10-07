# Proyecto ENDIREH: violencia contra las mujeres

Prácticas 3 y 4 de Almacenes y Minería de Datos. El proyecto reutiliza el conjunto
consolidado de ENDIREH 2021 para limpieza, EDA, medidas de localización y
variabilidad, heterogeneidad y concentración territorial, con Polars.

## Datos y alcance

Fuente: Encuesta Nacional sobre la Dinámica de las Relaciones en los Hogares
(ENDIREH) 2021, INEGI. Se utiliza el consolidado proporcionado para el curso.
Los resultados no se presentan como estimaciones oficiales de INEGI.

Colocar el archivo original en `data/data-raw/endireh_2021.csv`. Los datos no
se versionan y deben entregarse por separado. El diagnóstico y la limpieza
leen el crudo sin modificarlo. Los análisis usan el archivo procesado.
Todas las rutas de datos y salidas están centralizadas en `config/rutas.py`.

La limpieza genera `data/data-processed/endireh_2021_limpio.parquet`
(formato de trabajo) y `data/data-processed/endireh_2021_limpio.csv`
(para el requisito de entrega). Se conservan edades faltantes para evitar
una imputación masiva. Hijos incluye ceros imputados con bandera de auditoría.
Esta decisión y sus limitaciones se explican en el reporte.

## Instalación

Python 3.12 o superior para las versiones fijadas de NumPy y SciPy.
El entorno completo se verificó en Python 3.13. Esto supera el mínimo
académico de Python 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Polars realiza carga y agrupación. NumPy implementa los índices y ponderación.
SciPy calcula el CV muestral. Matplotlib/Seaborn producen gráficos. ReportLab
genera el PDF. Jupyter combina el
análisis con narrativa. Las funciones propias están en `src/analysis/indices.py`.

## Ejecución

Desde la raíz, con el entorno activado:

```bash
python -m src.cleaning.diagnostico
python -m src.cleaning.limpiar_datos
python -m src.reporting.practica4
```

El diagnóstico es de solo lectura. La limpieza genera los dos formatos
procesados. El último comando recalcula todas las figuras y tablas y produce
el reporte ejecutivo en `output/pdf/AyMD_Reporte_P4.pdf`.

También se pueden ejecutar los análisis por separado:

```bash
python -m src.visualization.eda
python -m src.analysis.medidas_descriptivas
python -m src.analysis.medidas_heterogeneidad
python -m src.analysis.medidas_concentracion
python -m src.analysis.gini_vs_entropia
```

## Notebooks y entregables

- `notebooks/01_eda.ipynb`: exploración y calidad de los datos.
- `notebooks/02_medidas_localizacion.ipynb`: medias, mediana simple y ponderada, modas y cuantiles.
- `notebooks/03_medidas_variabilidad.ipynb`: rango, varianza, desviación, CV e IQR.
- `notebooks/04_medidas_heterogeneidad.ipynb`: IQV, Gini-Simpson y Shannon.
- `notebooks/05_medidas_concentracion.ipynb`: Lorenz, Gini y prevalencia territorial.
- `notebooks/06_gini_vs_entropia.ipynb`: ambos índices sobre estado civil.
- `notebooks/reporte_p4.ipynb`: tabla integradora, figuras, decisiones y cuestionario.
- `reports/figures/`: gráficos generados.
- `output/pdf/AyMD_Reporte_P4.pdf`: reporte para entrega.

Abrir los notebooks con `jupyter notebook` y ejecutarlos de principio a fin.
El generador del PDF y los notebooks usan las mismas funciones de cálculo.
Si cambian los datos, volver a ejecutar los notebooks y actualizar las
interpretaciones numéricas antes de entregar. El reporte integrador y el PDF
calculan automáticamente su tabla y narrativa principal.

## Decisiones de análisis

Media y mediana ponderadas usan `factor_expansion`. La mediana ponderada es
el primer valor cuyo peso acumulado alcanza el 50%, sin interpolación.
La variabilidad, los cuantiles simples, los gráficos demográficos y la
heterogeneidad describen la muestra sin ponderación. En heterogeneidad se
muestra una representación por variable, evitando duplicar códigos y etiquetas.

Lorenz usa cantidades de casos ponderados por entidad. La prevalencia territorial
usa la población representada de cada entidad como denominador. La síntesis
calcula Gini y Shannon sobre las mismas frecuencias, tanto muestrales como
ponderadas. La diferencia entre entropías se debe a la ponderación.

La concentración de cantidades no equivale a mayor riesgo o prevalencia.
La diversidad no identifica por sí sola perfiles vulnerables ni demuestra
subregistro. No se realizan inferencias causales ni se calculan intervalos
sin el diseño muestral completo.

## Estructura

- `config/`: rutas centralizadas.
- `data/data-raw/`: originales sin modificar.
- `data/data-processed/`: datos limpios, CSV y Parquet.
- `data/data-input-model/` y `data/data-model/`: reservados para modelado.
- `src/cleaning/`, `src/analysis/`, `src/visualization/`: limpieza, cálculo y gráficos.
- `src/reporting/`: generación reproducible del reporte.
- `notebooks/`: análisis narrativo.
- `reports/` y `output/pdf/`: tablas, figuras y entregables.
