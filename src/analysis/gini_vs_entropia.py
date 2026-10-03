"""Medidas de localización y variabilidad para ENDIREH 2021."""

from __future__ import annotations

import numpy as np
import polars as pl

from config.rutas import ARCHIVO_ENDIREH_PROCESADO


VARIABLE = "estado_civil_desc" 

def cargar_datos() -> pl.DataFrame:
    """Carga el conjunto procesado y validado."""
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)



def comparar_entropia_y_gini_categorias(df: pl.DataFrame, col_cat: str, col_peso: str = "factor_expansion") -> dict:
    """
    Calcula la Entropía de Shannon y el Gini de frecuencias para una variable categórica ponderada.
    """
    # 1. Agrupar por la categoría, sumando el factor de expansión y ordenando de menor a mayor
    df_freq = (
        df.group_by(col_cat)
        .agg(
            freq = pl.col(col_peso).sum()
        )
        .sort("freq") # Orden ascendente necesario para el cálculo correcto del Gini de frecuencias
    )

    k = df_freq.height # Número de categorías
    if k <= 1:
        return {"variable": col_cat, "k": k, "Shannon": 0.0, "Gini_Frecuencias": 0.0}
    
    # 2. Calcular proporciones (p_i) para la Entropía de Shannon
    total_freq = df_freq["freq"].sum()
    
    df_metrics = df_freq.with_columns(
        p = pl.col("freq") / total_freq,
        # Creamos un índice de posición (1 hasta k) para la fórmula de Gini
        i = pl.int_range(1, k + 1, eager=False)
    )
    
    # 3. Cálculo de la Entropía de Shannon (base 2)
    # H = - sum(p * log2(p))
    shannon_val = df_metrics.select(
        (-pl.col("p") * pl.col("p").log(base=2)).sum()
    ).item()
    
    # 4. Cálculo del Coeficiente de Gini aplicado a las frecuencias de las categorías
    # Fórmula: G = [ (2 * sum(i * freq_i)) / (k * sum(freq_i)) ] - [ (k + 1) / k ]
    freqs = df_metrics["freq"].to_numpy()
    i_vals = np.arange(1, k + 1)
    
    suma_ponderada_i = np.sum(i_vals * freqs)
    gini_freq = (2 * suma_ponderada_i) / (k * total_freq) - ((k + 1) / k)
    
    return {
        "variable": col_cat,
        "k_categorias": k,
        "Entropia_Shannon": round(shannon_val, 4),
        "Gini_Frecuencias": round(gini_freq, 4)
    }


def main()-> None:
    df = cargar_datos()
    resultado_comparacion = comparar_entropia_y_gini_categorias(df,VARIABLE)
    for k, v in resultado_comparacion.items():
        print(f"  - {k}: {v}")



if __name__ == "__main__":
    main()
