"""Medidas de localización y variabilidad para ENDIREH 2021."""

from __future__ import annotations

import numpy as np
import polars as pl

from config.rutas import ARCHIVO_ENDIREH_PROCESADO


VARIABLES_CUALITATIVAS = [
    "nivel_escolaridad", 
    "estado_civil_id", "estado_civil_desc", 
    "estrato_socioeconomico", 
    "pareja_trabaja_id", "pareja_trabaja_desc", 
    "dinero_propio_id", "dinero_propio_desc", 
    "apoyo_gobierno_id", "apoyo_gobierno_desc", 
    "tiene_ahorros_id", "tiene_ahorros_desc", 
    "propietaria_vivienda_id", "propietaria_vivienda_desc", 
    "sufrio_violencia_pareja"
]

def cargar_datos() -> pl.DataFrame:
    """Carga el conjunto procesado y validado."""
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)



def calcular_indices_heterogeneidad(df: pl.DataFrame, col_name: str) -> dict:
    """
    Calcula el IQV, Gini-Simpson y Entropía de Shannon para una variable cualitativa en Polars.
    """
    # 1. Calcular frecuencias absolutas y relativas (p_i)
    df_freq = (
        df.group_by(col_name)
        .len(name="n")
        .with_columns(
            (pl.col("n") / pl.sum("n")).alias("p")
        )
    )
    
    k = df_freq.height  # Número de categorías (k)
    
    if k <= 1:
        return {
            "k_categorias": k,
            "Gini_Simpson": 0.0,
            "IQV": 0.0,
            "Shannon_Entropy": 0.0
        }
    
    # 2. Calcular las métricas usando expresiones de Polars
    metricas = df_freq.select(
        sum_p_sq = (pl.col("p") ** 2).sum(),
        # Usamos logaritmo base 2 para Shannon 
        shannon = (-pl.col("p") * pl.col("p").log(base=2)).sum()
    ).row(0, named=True)
    
    sum_p_sq = metricas["sum_p_sq"]
    shannon_val = metricas["shannon"]
    
    # Índice de Gini-Simpson
    gini_simpson = 1 - sum_p_sq
    
    # Índice de Variación Cualitativa (IQV)
    iqv = (k / (k - 1)) * gini_simpson
    
    return {
        "k_categorias": k,
        "Gini_Simpson": round(gini_simpson, 4),
        "IQV": round(iqv, 4),
        "Shannon_Entropy": round(shannon_val, 4)
    }

def medidas_heterogeneidad(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula medidas de heterogeneidad para todas las variables cualitativas."""
    filas: list[dict[str, object]] = []
    for variable in VARIABLES_CUALITATIVAS:
        indices = calcular_indices_heterogeneidad(df, variable)
        filas.append(
            {
                "variable": variable,
                "k_categorias": indices["k_categorias"],
                "Gini_Simpson": indices["Gini_Simpson"],
                "IQV": indices["IQV"],
                "Shannon_Entropy": indices["Shannon_Entropy"]
            }
        )
    return pl.DataFrame(filas)

def main()-> None:
    df = cargar_datos()
    heterogeneidad = medidas_heterogeneidad(df)
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4):
        print("=== MEDIDAS DE heterogeneidad ===")
        print(heterogeneidad)


if __name__ == "__main__":
    main()
