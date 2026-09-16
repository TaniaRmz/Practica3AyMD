"""Limpieza reproducible del conjunto consolidado ENDIREH 2021."""

from __future__ import annotations

import polars as pl

from config.rutas import (
    ARCHIVO_ENDIREH_PROCESADO,
    ARCHIVO_ENDIREH_RAW,
    RUTA_DATA_PROCESSED,
)


CODIGOS_INGRESO_FALTANTE = [999_998, 999_999]
CODIGO_INGRESO_CENSURADO = 999_997


def corregir_mojibake(valor: str | None) -> str | None:
    """Repara texto UTF-8 interpretado previamente como Latin-1."""
    if valor is None or "Ã" not in valor:
        return valor
    try:
        return valor.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return valor


def cargar_crudo() -> pl.DataFrame:
    """Carga el CSV original; las cadenas vacías se interpretan como nulos."""
    return pl.read_csv(
        ARCHIVO_ENDIREH_RAW,
        null_values=[""],
        infer_schema_length=10_000,
    )


def limpiar(df_crudo: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int | float]]:
    """Aplica las reglas aprobadas y devuelve los datos junto con su auditoría."""
    filas_entrada = df_crudo.height
    df = df_crudo.unique(maintain_order=True)
    duplicados_eliminados = filas_entrada - df.height

    # Normalización de edad: el descriptor oficial admite 9-97; 98 y 99
    # representan no recuerda/no especificado. Los valores 0-8 son inválidos.
    df = df.with_columns(
        (~pl.col("edad_primer_union").is_null()
         & ~pl.col("edad_primer_union").is_between(9, 97))
        .alias("edad_primer_union_corregida"),
        pl.when(pl.col("edad_primer_union").is_between(9, 97))
        .then(pl.col("edad_primer_union"))
        .otherwise(None)
        .cast(pl.Int16)
        .alias("edad_primer_union"),
    ).with_columns(
        pl.col("edad_primer_union").is_null().alias("edad_primer_union_faltante")
    )

    # Un nulo en num_hijos se interpreta como ausencia (cero), conservando una
    # bandera para distinguir valores informados de valores imputados.
    df = df.with_columns(
        pl.col("num_hijos").is_null().alias("num_hijos_imputado"),
        pl.col("num_hijos").fill_null(0).cast(pl.Int8).alias("num_hijos"),
    )

    # Los códigos 999998 y 999999 no son ingresos. El 999997 sí se conserva
    # como valor censurado y se marca para evitar interpretarlo como exacto.
    df = df.with_columns(
        pl.col("ingreso_pareja")
        .is_in(CODIGOS_INGRESO_FALTANTE)
        .fill_null(False)
        .alias("ingreso_pareja_codigo_faltante"),
        (pl.col("ingreso_pareja") == CODIGO_INGRESO_CENSURADO)
        .fill_null(False)
        .alias("ingreso_pareja_censurado"),
        pl.when(pl.col("ingreso_pareja").is_in(CODIGOS_INGRESO_FALTANTE))
        .then(None)
        .otherwise(pl.col("ingreso_pareja"))
        .alias("_ingreso_normalizado"),
    )

    # Las medianas de imputación excluyen el valor censurado para que este no
    # desplace artificialmente el centro de la distribución.
    ingreso_para_mediana = pl.when(
        (pl.col("_ingreso_normalizado") < CODIGO_INGRESO_CENSURADO)
        & (pl.col("pareja_trabaja_desc") == "Sí")
    ).then(pl.col("_ingreso_normalizado"))

    mediana_nacional = (
        df.filter(
            (pl.col("pareja_trabaja_desc") == "Sí")
            & pl.col("_ingreso_normalizado").is_not_null()
            & (pl.col("_ingreso_normalizado") < CODIGO_INGRESO_CENSURADO)
        )["_ingreso_normalizado"]
        .median()
    )
    if mediana_nacional is None:
        raise ValueError("No fue posible calcular la mediana nacional de ingreso.")

    df = df.with_columns(
        ingreso_para_mediana.median().over("cve_entidad").alias("_mediana_entidad")
    ).with_columns(
        (
            pl.col("_ingreso_normalizado").is_null()
            & pl.col("pareja_trabaja_desc").is_in(["Sí", "No"])
        ).alias("ingreso_pareja_imputado"),
        pl.when(
            pl.col("_ingreso_normalizado").is_null()
            & (pl.col("pareja_trabaja_desc") == "No")
        )
        .then(pl.lit("estructural_cero"))
        .when(
            pl.col("_ingreso_normalizado").is_null()
            & (pl.col("pareja_trabaja_desc") == "Sí")
        )
        .then(pl.lit("mediana_entidad"))
        .otherwise(pl.lit("sin_imputacion"))
        .alias("ingreso_pareja_tipo_imputacion"),
        pl.when(
            pl.col("_ingreso_normalizado").is_null()
            & (pl.col("pareja_trabaja_desc") == "No")
        )
        .then(pl.lit(0.0))
        .when(
            pl.col("_ingreso_normalizado").is_null()
            & (pl.col("pareja_trabaja_desc") == "Sí")
        )
        .then(pl.coalesce("_mediana_entidad", pl.lit(float(mediana_nacional))))
        .otherwise(pl.col("_ingreso_normalizado"))
        .alias("ingreso_pareja"),
    ).drop("_ingreso_normalizado", "_mediana_entidad")

    # Tipos compactos y semánticamente coherentes para códigos y conteos.
    df = df.with_columns(
        pl.col("cve_entidad").cast(pl.UInt8),
        pl.col("cve_municipio").cast(pl.UInt16),
        pl.col("estado_civil_id").cast(pl.UInt8),
        pl.col("estrato_socioeconomico").cast(pl.UInt8),
        pl.col("pareja_trabaja_id").cast(pl.UInt8),
        pl.col("dinero_propio_id").cast(pl.UInt8),
        pl.col("apoyo_gobierno_id").cast(pl.UInt8),
        pl.col("tiene_ahorros_id").cast(pl.UInt8),
        pl.col("propietaria_vivienda_id").cast(pl.UInt8),
        pl.col("sufrio_violencia_pareja").cast(pl.UInt8),
        pl.col("factor_expansion").cast(pl.UInt32),
        pl.col("anio_encuesta").cast(pl.UInt16),
    )

    # Corrige solamente cadenas con la marca típica de texto UTF-8 leído como
    # Latin-1 (p. ej. MÃ‰XICO); las cadenas ya correctas se conservan intactas.
    columnas_texto = [
        nombre for nombre, tipo in df.schema.items() if tipo == pl.String
    ]
    textos_corregidos = sum(
        int(df[columna].str.contains("Ã", literal=True).fill_null(False).sum())
        for columna in columnas_texto
    )
    df = df.with_columns(
        [
            pl.col(columna)
            .map_elements(corregir_mojibake, return_dtype=pl.String)
            .alias(columna)
            for columna in columnas_texto
        ]
    )

    # Al homologar códigos distintos que significan ausencia de información,
    # algunas filas antes diferentes se vuelven equivalentes. Se deduplican una
    # segunda vez para que el producto final no contenga repeticiones lógicas.
    filas_antes_deduplicacion_normalizada = df.height
    df = df.unique(maintain_order=True)
    duplicados_normalizados_eliminados = (
        filas_antes_deduplicacion_normalizada - df.height
    )

    auditoria: dict[str, int | float] = {
        "filas_entrada": filas_entrada,
        "duplicados_crudos_eliminados": duplicados_eliminados,
        "duplicados_normalizados_eliminados": duplicados_normalizados_eliminados,
        "filas_salida": df.height,
        "edades_corregidas": int(df["edad_primer_union_corregida"].sum()),
        "edades_faltantes_finales": int(df["edad_primer_union_faltante"].sum()),
        "num_hijos_imputados_cero": int(df["num_hijos_imputado"].sum()),
        "ingresos_codigo_faltante": int(df["ingreso_pareja_codigo_faltante"].sum()),
        "ingresos_censurados_conservados": int(df["ingreso_pareja_censurado"].sum()),
        "ingresos_imputados": int(df["ingreso_pareja_imputado"].sum()),
        "textos_corregidos": textos_corregidos,
        "mediana_nacional_respaldo": float(mediana_nacional),
    }
    return df, auditoria


def validar(df: pl.DataFrame, auditoria: dict[str, int | float]) -> None:
    """Falla de forma explícita si alguna regla esencial no se cumplió."""
    assert auditoria["filas_entrada"] == (
        auditoria["filas_salida"]
        + auditoria["duplicados_crudos_eliminados"]
        + auditoria["duplicados_normalizados_eliminados"]
    )
    assert not df.is_duplicated().any()
    assert df["edad_primer_union"].drop_nulls().is_between(9, 97).all()
    assert not df["ingreso_pareja"].is_null().any()
    assert not df["num_hijos"].is_null().any()
    assert not df["ingreso_pareja"].is_in(CODIGOS_INGRESO_FALTANTE).any()


def main() -> None:
    """Ejecuta, valida y guarda el conjunto limpio en formato Parquet."""
    df_limpio, auditoria = limpiar(cargar_crudo())
    validar(df_limpio, auditoria)

    RUTA_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_limpio.write_parquet(ARCHIVO_ENDIREH_PROCESADO, compression="zstd")

    print("=== AUDITORIA DE LIMPIEZA ===")
    for metrica, valor in auditoria.items():
        print(f"{metrica}: {valor:,}" if isinstance(valor, int) else f"{metrica}: {valor:,.2f}")
    print(f"columnas_salida: {df_limpio.width}")
    print(f"archivo_salida: {ARCHIVO_ENDIREH_PROCESADO}")


if __name__ == "__main__":
    main()
