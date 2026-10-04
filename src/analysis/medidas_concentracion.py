"""Concentración de cantidades de casos ponderados entre entidades."""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from config.rutas import ARCHIVO_ENDIREH_PROCESADO, RUTA_FIGURAS
from src.analysis.indices import curva_lorenz, coeficiente_gini

COLUM_ENTIDAD = "nom_entidad"


def cargar_datos() -> pl.DataFrame:
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)


def calcular_indices_gini(df: pl.DataFrame, column: str = COLUM_ENTIDAD,
                         show: bool = False) -> dict:
    tabla = (
        df.group_by(column)
        .agg((pl.col("sufrio_violencia_pareja").cast(pl.Float64)
              * pl.col("factor_expansion").cast(pl.Float64)).sum().alias("total_violencia"))
        .sort("total_violencia")
    )
    x, y = curva_lorenz(tabla["total_violencia"].to_numpy())
    gini = coeficiente_gini(tabla["total_violencia"].to_numpy())
    area = float(np.trapezoid(y, x))
    if not np.isclose(gini, 1 - 2 * area):
        raise ValueError("El Gini y el área de Lorenz no coinciden.")
    tabla = tabla.with_columns(pl.Series("p_entidades", x[1:]),
                               pl.Series("p_violencia", y[1:]))
    graficar_curva_lorenz(x, y, gini, show)
    return {"gini": gini, "area_lorenz": area, "tabla": tabla,
            "x_lorenz": x, "y_lorenz": y}


def graficar_curva_lorenz(x, y, coeficiente_gini, show=False) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, label=f"Lorenz (Gini = {coeficiente_gini:.4f})", color="#255F85", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Igual cantidad de casos por entidad")
    ax.set(title="Concentración territorial de casos de violencia de pareja",
           xlabel="Proporción acumulada de entidades",
           ylabel="Proporción acumulada de casos ponderados", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIGURAS / "08_curva_lorenz.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    resultado = calcular_indices_gini(cargar_datos())
    print(f"Gini: {resultado['gini']:.4f}. área Lorenz: {resultado['area_lorenz']:.4f}")
    with pl.Config(tbl_rows=32):
        print(resultado["tabla"])


if __name__ == "__main__":
    main()
