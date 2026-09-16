"""Medidas de localización y variabilidad para ENDIREH 2021."""

from __future__ import annotations

import numpy as np
import polars as pl

from config.rutas import ARCHIVO_ENDIREH_PROCESADO


VARIABLES_CUANTITATIVAS = ["edad_primer_union", "num_hijos"]


def cargar_datos() -> pl.DataFrame:
    """Carga el conjunto procesado y validado."""
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)


def media_ponderada(
    df: pl.DataFrame,
    variable: str,
    peso: str = "factor_expansion",
) -> float:
    """Calcula la media ponderada mediante NumPy usando pares completos."""
    completos = df.select(variable, peso).drop_nulls()
    valores = completos[variable].to_numpy()
    pesos = completos[peso].to_numpy()
    if valores.size == 0 or pesos.sum() == 0:
        return float("nan")
    return float(np.average(valores, weights=pesos))


def formato_moda(serie: pl.Series) -> str:
    """Representa una o varias modas sin descartar empates."""
    modas = serie.drop_nulls().mode().sort().to_list()
    return ", ".join(f"{float(valor):g}" for valor in modas)


def medidas_localizacion(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula localización simple y media ponderada por variable."""
    filas: list[dict[str, object]] = []
    for variable in VARIABLES_CUANTITATIVAS:
        serie = df[variable].drop_nulls()
        filas.append(
            {
                "variable": variable,
                "n_validos": serie.len(),
                "media_simple": serie.mean(),
                "media_ponderada": media_ponderada(df, variable),
                "mediana_q2": serie.median(),
                "moda": formato_moda(serie),
                "p10": serie.quantile(0.10, interpolation="linear"),
                "q1": serie.quantile(0.25, interpolation="linear"),
                "q3": serie.quantile(0.75, interpolation="linear"),
                "p90": serie.quantile(0.90, interpolation="linear"),
            }
        )
    return pl.DataFrame(filas)


def medidas_variabilidad(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula dispersión muestral para cada variable cuantitativa."""
    filas: list[dict[str, object]] = []
    for variable in VARIABLES_CUANTITATIVAS:
        serie = df[variable].drop_nulls()
        media = float(serie.mean())
        minimo = float(serie.min())
        maximo = float(serie.max())
        desviacion = float(serie.std(ddof=1))
        q1 = float(serie.quantile(0.25, interpolation="linear"))
        q3 = float(serie.quantile(0.75, interpolation="linear"))
        filas.append(
            {
                "variable": variable,
                "n_validos": serie.len(),
                "minimo": minimo,
                "maximo": maximo,
                "rango": maximo - minimo,
                "varianza_muestral": float(serie.var(ddof=1)),
                "desviacion_estandar": desviacion,
                "coeficiente_variacion_pct": desviacion / media * 100,
                "iqr": q3 - q1,
            }
        )
    return pl.DataFrame(filas)


def comparacion_por_violencia(df: pl.DataFrame) -> pl.DataFrame:
    """Compara localización y dispersión entre los grupos 0 y 1."""
    filas: list[dict[str, object]] = []
    for variable in VARIABLES_CUANTITATIVAS:
        for violencia in [0, 1]:
            grupo = df.filter(pl.col("sufrio_violencia_pareja") == violencia)
            serie = grupo[variable].drop_nulls()
            media = float(serie.mean())
            desviacion = float(serie.std(ddof=1))
            q1 = float(serie.quantile(0.25, interpolation="linear"))
            q3 = float(serie.quantile(0.75, interpolation="linear"))
            filas.append(
                {
                    "variable": variable,
                    "reporto_violencia": "Sí" if violencia == 1 else "No",
                    "n_validos": serie.len(),
                    "media_simple": media,
                    "media_ponderada": media_ponderada(grupo, variable),
                    "mediana": float(serie.median()),
                    "rango": float(serie.max()) - float(serie.min()),
                    "varianza_muestral": float(serie.var(ddof=1)),
                    "desviacion_estandar": desviacion,
                    "coeficiente_variacion_pct": desviacion / media * 100,
                    "iqr": q3 - q1,
                }
            )
    return pl.DataFrame(filas)


def validar_resultados(
    localizacion: pl.DataFrame,
    variabilidad: pl.DataFrame,
    comparacion: pl.DataFrame,
) -> None:
    """Comprueba que las tablas sean finitas y tengan el tamaño esperado."""
    assert localizacion.height == len(VARIABLES_CUANTITATIVAS)
    assert variabilidad.height == len(VARIABLES_CUANTITATIVAS)
    assert comparacion.height == len(VARIABLES_CUANTITATIVAS) * 2
    columnas_numericas = [
        columna
        for tabla in [localizacion, variabilidad, comparacion]
        for columna, tipo in tabla.schema.items()
        if tipo.is_numeric()
    ]
    assert columnas_numericas
    for tabla in [localizacion, variabilidad, comparacion]:
        for columna, tipo in tabla.schema.items():
            if tipo.is_float():
                assert not tabla[columna].is_nan().any()


def main() -> None:
    """Calcula, valida e imprime todas las medidas descriptivas."""
    df = cargar_datos()
    localizacion = medidas_localizacion(df)
    variabilidad = medidas_variabilidad(df)
    comparacion = comparacion_por_violencia(df)
    validar_resultados(localizacion, variabilidad, comparacion)

    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4):
        print("=== MEDIDAS DE LOCALIZACION ===")
        print(localizacion)
        print("\n=== MEDIDAS DE VARIABILIDAD ===")
        print(variabilidad)
        print("\n=== COMPARACION POR REPORTE DE VIOLENCIA ===")
        print(comparacion)


if __name__ == "__main__":
    main()
