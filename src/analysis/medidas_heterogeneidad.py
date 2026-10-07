"""Diversidad de variables cualitativas en la muestra, sin ponderación."""

import polars as pl
import matplotlib.pyplot as plt
from config.rutas import ARCHIVO_ENDIREH_PROCESADO, RUTA_FIGURAS
from src.analysis.indices import indices_heterogeneidad

ETIQUETAS = {
    "nivel_escolaridad": "Escolaridad",
    "estado_civil_desc": "Estado civil",
    "estrato_socioeconomico": "Estrato socioeconómico",
    "pareja_trabaja_desc": "Trabajo de la pareja",
    "dinero_propio_desc": "Dinero propio",
    "apoyo_gobierno_desc": "Apoyo del gobierno",
    "tiene_ahorros_desc": "Ahorros",
    "propietaria_vivienda_desc": "Propiedad de vivienda",
    "sufrio_violencia_pareja": "Reporte de violencia de pareja",
}
VARIABLES_CUALITATIVAS = list(ETIQUETAS)


def cargar_datos() -> pl.DataFrame:
    return pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)


def calcular_indices_heterogeneidad(df: pl.DataFrame, col_name: str) -> dict:
    """Excluye nulos. los índices describen frecuencias muestrales positivas."""
    validos = df.filter(pl.col(col_name).is_not_null())
    frecuencias = validos.group_by(col_name).len(name="n")["n"].to_numpy()
    return indices_heterogeneidad(frecuencias)


def medidas_heterogeneidad(df: pl.DataFrame) -> pl.DataFrame:
    filas = [{"variable": variable, **calcular_indices_heterogeneidad(df, variable)}
             for variable in VARIABLES_CUALITATIVAS]
    return pl.DataFrame(filas).sort("IQV", descending=True)


def generar_grafica_iqv(df_resultado: pl.DataFrame, show=False) -> None:
    tabla = df_resultado.sort("IQV")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    barras = ax.barh([ETIQUETAS.get(v, v) for v in tabla["variable"]],
                    tabla["IQV"].to_list(), color="#255F85")
    ax.bar_label(barras, labels=[f"{v:.3f}" for v in tabla["IQV"]], padding=5, fontsize=9)
    ax.set(title="Diversidad de respuestas en la muestra",
           xlabel="IQV (0 = homogeneidad, 1 = máxima diversidad)", xlim=(0, 1.12))
    ax.grid(axis="x", linestyle=":", alpha=.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIGURAS / "07_barras_iqv.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    heterogeneidad = medidas_heterogeneidad(cargar_datos())
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4):
        print(heterogeneidad)
    generar_grafica_iqv(heterogeneidad)


if __name__ == "__main__":
    main()
