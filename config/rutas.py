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
RUTA_PDF = RUTA_PROYECTO / "output" / "pdf"
ARCHIVO_REPORTE_P4 = RUTA_PDF / "AyMD_Reporte_P4.pdf"
ARCHIVO_ENDIREH_PROCESADO_CSV = RUTA_DATA_PROCESSED / "endireh_2021_limpio.csv"

ARCHIVO_ENDIREH_RAW = RUTA_DATA_RAW / "endireh_2021.csv"
ARCHIVO_ENDIREH_PROCESADO = RUTA_DATA_PROCESSED / "endireh_2021_limpio.parquet"
