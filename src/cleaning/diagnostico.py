"""Diagnóstico reproducible del archivo crudo de ENDIREH 2021.

Este módulo es deliberadamente de solo lectura: no transforma ni exporta datos.
"""

from __future__ import annotations

import polars as pl

from config.rutas import ARCHIVO_ENDIREH_RAW


VARIABLES_CATEGORICAS = [
    "cve_entidad",
    "nom_entidad",
    "cve_municipio",
    "nom_municipio",
    "nivel_escolaridad",
    "estado_civil_id",
    "estado_civil_desc",
    "estrato_socioeconomico",
    "pareja_trabaja_id",
    "pareja_trabaja_desc",
    "dinero_propio_id",
    "dinero_propio_desc",
    "apoyo_gobierno_id",
    "apoyo_gobierno_desc",
    "tiene_ahorros_id",
    "tiene_ahorros_desc",
    "propietaria_vivienda_id",
    "propietaria_vivienda_desc",
    "sufrio_violencia_pareja",
    "anio_encuesta",
]

VARIABLES_NUMERICAS = [
    "edad_primer_union",
    "num_hijos",
    "ingreso_pareja",
    "factor_expansion",
]


def cargar_datos() -> pl.DataFrame:
    """Carga el CSV con cadenas vacías interpretadas como valores nulos."""
    return pl.read_csv(
        ARCHIVO_ENDIREH_RAW,
        null_values=[""],
        infer_schema_length=10_000,
    )


def tabla_nulos(df: pl.DataFrame) -> pl.DataFrame:
    """Devuelve conteo y porcentaje de nulos por columna."""
    return pl.DataFrame(
        {
            "columna": df.columns,
            "tipo_inferido": [str(tipo) for tipo in df.dtypes],
            "nulos": [df[columna].null_count() for columna in df.columns],
        }
    ).with_columns(
        (pl.col("nulos") / df.height * 100).round(2).alias("porcentaje_nulos")
    ).sort("porcentaje_nulos", descending=True)


def resumen_numerico(df: pl.DataFrame) -> pl.DataFrame:
    """Resume rango y cuantiles de las variables numéricas principales."""
    filas: list[dict[str, object]] = []
    for columna in VARIABLES_NUMERICAS:
        serie = df[columna].drop_nulls()
        filas.append(
            {
                "variable": columna,
                "observaciones": serie.len(),
                "minimo": serie.min(),
                "p01": serie.quantile(0.01),
                "q1": serie.quantile(0.25),
                "mediana": serie.median(),
                "q3": serie.quantile(0.75),
                "p99": serie.quantile(0.99),
                "maximo": serie.max(),
            }
        )
    return pl.DataFrame(filas)


def cardinalidad_categorica(df: pl.DataFrame) -> pl.DataFrame:
    """Cuenta valores distintos observados en variables categóricas."""
    return pl.DataFrame(
        {
            "variable": VARIABLES_CATEGORICAS,
            "valores_distintos_sin_nulos": [
                df[columna].drop_nulls().n_unique()
                for columna in VARIABLES_CATEGORICAS
            ],
        }
    )


def inconsistencias_id_descripcion(df: pl.DataFrame) -> pl.DataFrame:
    """Cuenta asociaciones ID-descripción distintas para revisar su consistencia."""
    pares = [
        ("estado_civil_id", "estado_civil_desc"),
        ("pareja_trabaja_id", "pareja_trabaja_desc"),
        ("dinero_propio_id", "dinero_propio_desc"),
        ("apoyo_gobierno_id", "apoyo_gobierno_desc"),
        ("tiene_ahorros_id", "tiene_ahorros_desc"),
        ("propietaria_vivienda_id", "propietaria_vivienda_desc"),
    ]
    resultados = []
    for columna_id, columna_desc in pares:
        asociaciones = (
            df.select(columna_id, columna_desc)
            .drop_nulls()
            .unique()
            .group_by(columna_id)
            .agg(pl.col(columna_desc).n_unique().alias("descripciones_por_id"))
            .filter(pl.col("descripciones_por_id") > 1)
        )
        resultados.append(
            {
                "par": f"{columna_id} / {columna_desc}",
                "ids_con_multiples_descripciones": asociaciones.height,
            }
        )
    return pl.DataFrame(resultados)


def main() -> None:
    """Imprime un perfil compacto y auditable del archivo original."""
    df = cargar_datos()

    print("=== DIMENSIONES ===")
    print(f"filas: {df.height:,}")
    print(f"columnas: {df.width}")
    print(f"filas duplicadas completas: {df.is_duplicated().sum():,}")

    print("\n=== NULOS Y TIPOS ===")
    print(tabla_nulos(df))

    print("\n=== RESUMEN NUMERICO ===")
    print(resumen_numerico(df))

    print("\n=== CARDINALIDAD CATEGORICA ===")
    print(cardinalidad_categorica(df))

    print("\n=== FRECUENCIA DE VIOLENCIA DE PAREJA ===")
    print(
        df.group_by("sufrio_violencia_pareja")
        .len(name="filas")
        .with_columns(
            (pl.col("filas") / df.height * 100).round(2).alias("porcentaje")
        )
        .sort("sufrio_violencia_pareja")
    )

    print("\n=== CONSISTENCIA ID-DESCRIPCION ===")
    print(inconsistencias_id_descripcion(df))


if __name__ == "__main__":
    main()
