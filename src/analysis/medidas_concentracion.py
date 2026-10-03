"""Medidas de localización y variabilidad para ENDIREH 2021."""

from __future__ import annotations
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from config.rutas import ARCHIVO_ENDIREH_PROCESADO,RUTA_FIGURAS

COLUM_ENTIDAD = "nom_entidad" 


def cargar_datos() -> pl.DataFrame:
    """Carga el conjunto procesado y validado."""
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)


def calcular_indices_gini(df: pl.DataFrame, column: str, show = True) -> tuple[list[float], list[float]]:
    """ Calcula el Coeficiente de Gini para una variable categórica
        ponderada por el factor de expansión.
    """
    # 1. Agrupar y calcular casos ponderados de violencia por entidad
    df_gini = (
        df.with_columns(
            # Calcular casos ponderados de violencia
            violencia_ponderada = pl.col("sufrio_violencia_pareja") * pl.col("factor_expansion")
        )
        .group_by(column)
        .agg(
            total_violencia = pl.col("violencia_ponderada").sum()
        )
        .sort("total_violencia") # Ordenar de menor a mayor para la Curva de Lorenz
    )

    # 2. Calcular las proporciones acumuladas para la Curva de Lorenz
    n_entidades = df_gini.height


    df_lorenz = df_gini.with_columns(
        # Proporción acumulada de entidades (eje X)
        p_entidades = (pl.arange(1, n_entidades + 1) / n_entidades),
        
        # Proporción acumulada de la variable de violencia (eje Y)
        p_violencia = pl.col("total_violencia").cum_sum() / pl.col("total_violencia").sum()
    )

    # Añadir el punto inicial (0, 0) para graficar correctamente desde el origen
    x_lorenz = [0.0] + df_lorenz["p_entidades"].to_list()
    y_lorenz = [0.0] + df_lorenz["p_violencia"].to_list()

    # 3. Cálculo del Coeficiente de Gini (usando la regla trapezoidal del área bajo la curva)
    # Gini = 1 - 2 * (Área bajo la curva de Lorenz)
    area_bajo_lorenz = np.trapezoid(y_lorenz, x_lorenz)
    coeficiente_gini = 1 - (2 * area_bajo_lorenz)
    print(f"Coeficiente de Gini para {column}: {coeficiente_gini:.4f}")
    print (f"Área bajo la curva de Lorenz: {area_bajo_lorenz:.4f}")
    print ( df_lorenz)
    graficar_curva_lorenz(x_lorenz, y_lorenz, coeficiente_gini, show = show)

def graficar_curva_lorenz(x: list[float], y: list[float], coeficiente_gini, show = False) -> None:

    # 4. Graficar la Curva de Lorenz
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"Curva de Lorenz (Gini = {coeficiente_gini:.3f})", color="b", lw=2)
    # Línea de equidistribución perfecta (diagonal de 45 grados)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Equidistribución perfecta")

    plt.title("Concentración Territorial de Violencia de Pareja")
    plt.xlabel("Proporción acumulada de Entidades Federativas")
    plt.ylabel("Proporción acumulada de Casos Ponderados")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.title("Concentración Territorial de Violencia de Pareja")
    plt.xlabel("Proporción acumulada de Entidades Federativas")
    plt.ylabel("Proporción acumulada de Casos Ponderados")
    plt.legend()
    if show:
        plt.show()
    plt.grid(True, linestyle=":", alpha=0.6)
    guardar(plt, "08_curva_lorenz.png")
    plt.close() 

def guardar(figura: plt.Figure, nombre: str) -> None:
    """Guarda y cierra una figura, evitando estado global acumulado."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(RUTA_FIGURAS / nombre, bbox_inches="tight", facecolor="white")


def main()-> None:
    df = cargar_datos()
    calcular_indices_gini(df,COLUM_ENTIDAD, show = False)




if __name__ == "__main__":
    main()
