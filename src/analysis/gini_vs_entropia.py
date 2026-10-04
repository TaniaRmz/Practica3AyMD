"""Gini y Shannon sobre las mismas frecuencias de estado civil."""

import polars as pl
from config.rutas import ARCHIVO_ENDIREH_PROCESADO
from src.analysis.indices import coeficiente_gini, indices_heterogeneidad

VARIABLE = "estado_civil_id"


def cargar_datos() -> pl.DataFrame:
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)


def comparar_entropia_y_gini_categorias(df: pl.DataFrame, col_cat: str,
                                       col_peso: str | None = "factor_expansion") -> dict:
    validos = df.filter(pl.col(col_cat).is_not_null())
    if col_peso is None:
        tabla = validos.group_by(col_cat).len(name="freq")
    else:
        tabla = validos.group_by(col_cat).agg(pl.col(col_peso).cast(pl.Float64).sum().alias("freq"))
    frecuencias = tabla["freq"].to_numpy()
    indices = indices_heterogeneidad(frecuencias)
    return {"variable": col_cat, "k_categorias": indices["k_categorias"],
            "Entropia_Shannon": indices["Shannon_Entropy"],
            "Gini_Frecuencias": coeficiente_gini(frecuencias),
            "ponderacion": col_peso or "sin ponderar"}


def main() -> None:
    df = cargar_datos()
    for peso in [None, "factor_expansion"]:
        print(comparar_entropia_y_gini_categorias(df, VARIABLE, peso))


if __name__ == "__main__":
    main()
