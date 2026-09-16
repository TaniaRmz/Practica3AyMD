"""Visualizaciones del análisis exploratorio de ENDIREH 2021."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "practica3-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config.rutas import ARCHIVO_ENDIREH_PROCESADO, RUTA_FIGURAS


COLOR_AZUL = "#255F85"
COLOR_NARANJA = "#D97941"
COLOR_GRIS = "#6B7280"


def configurar_estilo() -> None:
    """Define un estilo consistente y legible para todas las figuras."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def guardar(figura: plt.Figure, nombre: str) -> None:
    """Guarda y cierra una figura, evitando estado global acumulado."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(RUTA_FIGURAS / nombre, bbox_inches="tight", facecolor="white")
    plt.close(figura)


def grafica_faltantes(df: pl.DataFrame) -> None:
    faltantes = (
        pl.DataFrame(
            {
                "variable": df.columns,
                "porcentaje": [df[c].null_count() / df.height * 100 for c in df.columns],
            }
        )
        .filter(pl.col("porcentaje") > 0)
        .sort("porcentaje")
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(faltantes["variable"].to_list(), faltantes["porcentaje"].to_list(), color=COLOR_AZUL)
    ax.set_title("Datos faltantes después del preprocesamiento")
    ax.set_xlabel("Porcentaje de registros")
    ax.set_xlim(0, 100)
    for i, valor in enumerate(faltantes["porcentaje"]):
        ax.text(valor + 1, i, f"{valor:.1f}%", va="center")
    guardar(fig, "01_faltantes.png")


def grafica_edad(df: pl.DataFrame) -> None:
    edades = df["edad_primer_union"].drop_nulls().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(8.5, 98.5, 2)
    ax.hist(edades, bins=bins, color=COLOR_AZUL, edgecolor="white")
    ax.set_title("Edad reportada al contraer la primera unión")
    ax.set_xlabel("Edad (años cumplidos)")
    ax.set_ylabel("Registros válidos")
    ax.text(0.99, 0.95, f"n = {len(edades):,}", transform=ax.transAxes, ha="right", va="top")
    guardar(fig, "02_distribucion_edad.png")


def grafica_hijos(df: pl.DataFrame) -> None:
    conteos = df.group_by("num_hijos").len().sort("num_hijos")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(
        [str(v) for v in conteos["num_hijos"]],
        conteos["len"].to_list(),
        color=COLOR_AZUL,
    )
    ax.set_title("Distribución del número de hijas e hijos")
    ax.set_xlabel("Número de hijas e hijos")
    ax.set_ylabel("Registros")
    ax.bar_label(barras, labels=[f"{v:,}" for v in conteos["len"]], padding=3, fontsize=8)
    ax.text(
        0.5,
        1.01,
        "El valor 0 incluye imputaciones documentadas",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=COLOR_GRIS,
        fontsize=8,
    )
    guardar(fig, "03_distribucion_hijos.png")


def grafica_violencia(df: pl.DataFrame) -> None:
    resumen = (
        df.group_by("sufrio_violencia_pareja")
        .agg(pl.len().alias("n"), pl.col("factor_expansion").sum().alias("peso"))
        .with_columns(
            (pl.col("n") / pl.col("n").sum() * 100).alias("muestra"),
            (pl.col("peso") / pl.col("peso").sum() * 100).alias("ponderado"),
        )
        .sort("sufrio_violencia_pareja")
    )
    x = np.arange(2)
    ancho = 0.34
    fig, ax = plt.subplots(figsize=(7, 4.6))
    b1 = ax.bar(x - ancho / 2, resumen["muestra"], ancho, label="Muestra", color=COLOR_AZUL)
    b2 = ax.bar(x + ancho / 2, resumen["ponderado"], ancho, label="Ponderado", color=COLOR_NARANJA)
    ax.set_title("Reporte de violencia de pareja")
    ax.set_ylabel("Porcentaje")
    ax.set_xticks(x, ["No reportó", "Sí reportó"])
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.bar_label(b1, fmt="%.1f%%", padding=3)
    ax.bar_label(b2, fmt="%.1f%%", padding=3)
    guardar(fig, "04_violencia_muestra_ponderada.png")


def grafica_edad_por_violencia(df: pl.DataFrame) -> None:
    grupos = [
        df.filter(pl.col("sufrio_violencia_pareja") == valor)["edad_primer_union"]
        .drop_nulls()
        .to_numpy()
        for valor in [0, 1]
    ]
    fig, ax = plt.subplots(figsize=(7, 4.7))
    cajas = ax.boxplot(grupos, tick_labels=["No reportó", "Sí reportó"], patch_artist=True)
    for caja, color in zip(cajas["boxes"], [COLOR_AZUL, COLOR_NARANJA], strict=True):
        caja.set_facecolor(color)
        caja.set_alpha(0.8)
    ax.set_title("Edad de primera unión según reporte de violencia")
    ax.set_ylabel("Edad (años cumplidos)")
    ax.text(
        0.99,
        0.95,
        f"Casos válidos: {len(grupos[0]):,} y {len(grupos[1]):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLOR_GRIS,
        fontsize=8,
    )
    guardar(fig, "05_edad_por_violencia.png")


def grafica_entidades(df: pl.DataFrame) -> None:
    entidades = (
        df.group_by("nom_entidad")
        .agg(
            (
                (pl.col("sufrio_violencia_pareja") * pl.col("factor_expansion")).sum()
                / pl.col("factor_expansion").sum()
                * 100
            ).alias("porcentaje_ponderado")
        )
        .sort("porcentaje_ponderado")
    )
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(
        entidades["nom_entidad"].to_list(),
        entidades["porcentaje_ponderado"].to_list(),
        color=COLOR_AZUL,
    )
    ax.set_title("Reporte ponderado de violencia de pareja por entidad")
    ax.set_xlabel("Porcentaje ponderado")
    ax.set_ylabel("")
    guardar(fig, "06_violencia_por_entidad.png")


def main() -> None:
    """Genera todas las figuras del EDA a partir del Parquet limpio."""
    configurar_estilo()
    df = pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)
    grafica_faltantes(df)
    grafica_edad(df)
    grafica_hijos(df)
    grafica_violencia(df)
    grafica_edad_por_violencia(df)
    grafica_entidades(df)
    print(f"Se generaron 6 figuras en: {RUTA_FIGURAS}")


if __name__ == "__main__":
    main()
