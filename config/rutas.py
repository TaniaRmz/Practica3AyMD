"""Rutas estáticas del proyecto ENDIREH."""

from pathlib import Path


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATA = RUTA_PROYECTO / "data"
RUTA_DATA_RAW = RUTA_DATA / "data-raw"
RUTA_DATA_PROCESSED = RUTA_DATA / "data-processed"
RUTA_DATA_INPUT_MODEL = RUTA_DATA / "data-input-model"
RUTA_DATA_MODEL = RUTA_DATA / "data-model"
RUTA_REPORTS = RUTA_PROYECTO / "reports"
RUTA_FIGURAS = RUTA_REPORTS / "figures"

ARCHIVO_ENDIREH_RAW = RUTA_DATA_RAW / "endireh_2021.csv"
ARCHIVO_ENDIREH_PROCESADO = RUTA_DATA_PROCESSED / "endireh_2021_limpio.parquet"
