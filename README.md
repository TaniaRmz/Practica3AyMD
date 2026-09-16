# Proyecto ENDIREH: violencia contra las mujeres

Práctica 3 del curso de Almacenes y Minería de Datos. El proyecto organiza y analiza
un conjunto consolidado de la ENDIREH 2021 con Polars. Incluye carga reproducible,
preprocesamiento, análisis exploratorio, medidas descriptivas y visualizaciones.

## Fuente de datos

Encuesta Nacional sobre la Dinámica de las Relaciones en los Hogares (ENDIREH) 2021,
del Instituto Nacional de Estadística y Geografía (INEGI).

El archivo original debe colocarse en:
`data/data-raw/endireh_2021.csv`.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

Desde la raíz del proyecto y con el entorno virtual activado:

```bash
python -m src.cleaning.diagnostico
```

Este comando genera un perfil de solo lectura del CSV crudo; no modifica ni exporta
los datos.

Para ejecutar la limpieza aprobada y producir el archivo procesado:

```bash
python -m src.cleaning.limpiar_datos
```

El resultado se guarda como `data/data-processed/endireh_2021_limpio.parquet`.
El CSV ubicado en `data/data-raw/` nunca se modifica.

Para generar las visualizaciones del EDA:

```bash
python -m src.visualization.eda
```

El notebook documentado se encuentra en `notebooks/01_eda.ipynb`.

Para calcular las medidas descriptivas:

```bash
python -m src.analysis.medidas_descriptivas
```

La interpretación completa se encuentra en
`notebooks/02_medidas_descriptivas.ipynb`.

## Estructura

- `config/`: rutas centralizadas del proyecto.
- `data/data-raw/`: datos originales sin modificar.
- `data/data-processed/`: datos limpios y con tipos corregidos.
- `data/data-input-model/`: datos transformados para modelado.
- `data/data-model/`: salidas de modelos.
- `src/cleaning/`: scripts de limpieza.
- `src/visualization/`: scripts de EDA y gráficas.
- `src/models/`: scripts de modelado.
- `notebooks/`: análisis exploratorio documentado.
