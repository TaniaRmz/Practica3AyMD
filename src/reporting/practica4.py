"""Tabla, narrativa y PDF generados desde los resultados recalculados."""

from html import escape
import polars as pl
from config.rutas import (ARCHIVO_ENDIREH_PROCESADO, ARCHIVO_REPORTE_P4,
                          RUTA_FIGURAS, RUTA_PDF)
from src.analysis.medidas_descriptivas import (medidas_localizacion, medidas_variabilidad,
                                              comparacion_por_violencia, validar_resultados)
from src.analysis.medidas_heterogeneidad import medidas_heterogeneidad, generar_grafica_iqv
from src.analysis.medidas_concentracion import calcular_indices_gini
from src.analysis.gini_vs_entropia import comparar_entropia_y_gini_categorias
from src.visualization.eda import main as generar_eda


def resultados() -> dict:
    df = pl.read_parquet(ARCHIVO_ENDIREH_PROCESADO)
    localizacion, variabilidad = medidas_localizacion(df), medidas_variabilidad(df)
    comparacion = comparacion_por_violencia(df)
    validar_resultados(localizacion, variabilidad, comparacion)
    heterogeneidad = medidas_heterogeneidad(df)
    concentracion = calcular_indices_gini(df)
    sintesis = pl.DataFrame([comparar_entropia_y_gini_categorias(df, "estado_civil_id", peso)
                            for peso in [None, "factor_expansion"]])
    prevalencia = df.group_by("nom_entidad").agg(
        (pl.col("sufrio_violencia_pareja").cast(pl.Float64)
         * pl.col("factor_expansion").cast(pl.Float64)).sum().alias("casos_ponderados"),
        pl.col("factor_expansion").cast(pl.Float64).sum().alias("poblacion_representada")
    ).with_columns((100 * pl.col("casos_ponderados") / pl.col("poblacion_representada"))
                   .alias("porcentaje_ponderado")).sort("porcentaje_ponderado")
    return dict(df=df, localizacion=localizacion, variabilidad=variabilidad,
                comparacion=comparacion, heterogeneidad=heterogeneidad,
                concentracion=concentracion, sintesis=sintesis, prevalencia=prevalencia)


def resumen(r: dict) -> list[dict]:
    loc, var = r["localizacion"].to_dicts(), r["variabilidad"].to_dicts()
    h = {f["variable"]: f for f in r["heterogeneidad"].to_dicts()}
    g, s = r["concentracion"], r["sintesis"].to_dicts()[1]
    return [
        dict(familia="Localización", variables="Edad de primera unión y número de hijos",
             valor=". ".join(f"{f['variable']}: media pond. {f['media_ponderada']:.2f}, mediana pond. {f['mediana_ponderada']:.0f}" for f in loc),
             interpretacion="La edad usa los datos válidos. Hijos incluye los valores rellenados con cero."),
        dict(familia="Variabilidad", variables="Edad de primera unión y número de hijos",
             valor=". ".join(f"{f['variable']}: CV {f['coeficiente_variacion_pct']:.2f}%, IQR {f['iqr']:.0f}" for f in var),
             interpretacion="El IQR de edad es 13 años y el de hijos es 2. Ambos se calcularon sin ponderar."),
        dict(familia="Heterogeneidad", variables="Violencia de pareja y estado civil",
             valor=f"Violencia: H {h['sufrio_violencia_pareja']['Shannon_Entropy']:.4f}, IQV {h['sufrio_violencia_pareja']['IQV']:.4f}. Estado civil: H {h['estado_civil_desc']['Shannon_Entropy']:.4f}, IQV {h['estado_civil_desc']['IQV']:.4f}",
             interpretacion="Muestra cómo se reparten las respuestas. No permite saber si hay subregistro."),
        dict(familia="Concentración", variables="Casos ponderados por entidad",
             valor=f"Gini {g['gini']:.4f}, área Lorenz {g['area_lorenz']:.4f}",
             interpretacion="Los casos no se reparten por igual. También influye el tamaño de la población."),
        dict(familia="Gini vs. entropía", variables="Estado civil (6 categorías), ponderado",
             valor=f"H {s['Entropia_Shannon']:.4f} bits, Gini de frecuencias {s['Gini_Frecuencias']:.4f}",
             interpretacion="Shannon mide diversidad y Gini mide desigualdad entre las frecuencias."),
    ]


def tabla_markdown(r: dict) -> str:
    filas = ["| Familia | Variables | Valor obtenido | Interpretación |",
             "|---|---|---|---|"]
    filas += ["| " + " | ".join(f.values()) + " |" for f in resumen(r)]
    return "\n".join(filas)


def metodologia(r: dict) -> str:
    df = r["df"]
    faltantes = df["edad_primer_union"].null_count()
    imputados = int(df["num_hijos_imputado"].sum())
    return (
        f"Usamos los datos de ENDIREH 2021 que limpiamos en la práctica 3. El archivo tiene {df.height:,} registros "
        f"y {df.width} columnas. Trabajamos con Parquet y también guardamos una copia en CSV. "
        f"Falta la edad en {faltantes:,} registros ({100*faltantes/df.height:.2f}%). No la rellenamos porque "
        f"faltan demasiados datos. Los cálculos de edad usan los {df.height-faltantes:,} registros que sí tienen dato. "
        f"En hijos se rellenaron {imputados:,} valores con cero en la práctica 3. Esa decisión puede afectar los resultados. "
        "Elegimos edad e hijos para describir las características demográficas. Para heterogeneidad usamos "
        "escolaridad, estado civil y condiciones económicas. Para concentración agrupamos por entidad. "
        "Usamos Polars para cargar y agrupar los datos, NumPy para los índices y SciPy para el CV "
        "con stats.variation y ddof=1. Las gráficas se hicieron con Matplotlib y Seaborn. "
        "Las funciones de los índices están en src/analysis/indices.py. Los notebooks reúnen los cálculos "
        "y las interpretaciones. Git guarda los cambios del proyecto. "
        "La media y la mediana ponderadas usan factor_expansion. Para la mediana ordenamos los valores "
        "y tomamos el primero cuyo peso acumulado llega al 50%. La variabilidad y la heterogeneidad "
        "se calcularon sin ponderar para describir la muestra. Los casos por entidad y sus porcentajes "
        "sí usan el factor de expansión. En la comparación de Gini y Shannon calculamos una versión "
        "sin ponderar y otra ponderada para ver la diferencia."
    )


def sintesis_texto(r: dict) -> str:
    m, p = r["sintesis"].to_dicts()
    return (f"Para estado civil obtuvimos una entropía ponderada de {p['Entropia_Shannon']:.4f} bits "
            f"y un Gini de frecuencias de {p['Gini_Frecuencias']:.4f}. Shannon muestra qué tan repartidas "
            "están las respuestas entre las seis categorías. Su máximo es log2(6), aproximadamente 2.5850. "
            "El Gini muestra qué tan desiguales son las frecuencias de esas categorías. Los dos usan los "
            "mismos datos, pero responden preguntas distintas. "
            f"Sin ponderar, Shannon es {m['Entropia_Shannon']:.4f}. La diferencia se debe al factor de expansión. "
            "Estos índices no muestran la relación entre estado civil y violencia. Tampoco hay que confundir "
            "el Gini de concentración con Gini-Simpson, que mide diversidad.")


def cuestionario(r: dict) -> list[tuple[str, str]]:
    g = r["concentracion"]["gini"]
    return [
        ("1. ¿Qué muestran Lorenz y Gini?",
         f"Obtuvimos un Gini de {g:.4f} y la curva quedó por debajo de la diagonal. Esto muestra que "
         "los casos ponderados no se reparten por igual entre las entidades. Sin embargo, sus poblaciones "
         "son distintas. Tener más casos no significa necesariamente tener un porcentaje mayor de violencia. "
         "Por eso también revisamos el porcentaje de cada entidad. El Gini que calculamos resume la "
         "concentración entre todas las entidades, no es un índice para cada estado."),
        ("2. Limitaciones y mejoras concretas",
         "Una limitación es la cantidad de edades faltantes. Otra es haber rellenado hijos con cero, "
         "porque eso puede cambiar su distribución. También eliminamos duplicados sin tener un "
         "identificador de entrevista que permita comprobar cada caso. Las respuestas de la encuesta "
         "dependen de lo que las personas estén dispuestas a contar, pero estos cálculos no permiten "
         "saber cuánto subregistro hay. Como mejoras, revisaríamos los códigos y las preguntas del "
         "cuestionario y compararíamos los resultados de hijos con y sin los ceros imputados. "
         "También buscaríamos identificadores y variables del diseño muestral para revisar los duplicados "
         "y calcular intervalos de confianza. Para analizar municipios primero habría que revisar si "
         "hay suficientes datos y cuidar la privacidad de las personas."),
        ("3. ¿Cómo orientan decisiones posteriores?",
         "Las medidas de localización y variabilidad ayudan a revisar los valores extremos y las "
         "decisiones de limpieza. La heterogeneidad permite ver qué variables tienen categorías "
         "equilibradas y en cuáles predomina una respuesta. La concentración ayuda a ver dónde se "
         "acumulan más casos, pero para decidir dónde hacen falta recursos también hay que revisar "
         "los porcentajes y el tamaño de la población. Para elegir variables de un modelo necesitaríamos "
         "estudiar su relación con la violencia y validar sus resultados. Estas medidas no bastan "
         "para identificar causas o grupos con mayor riesgo."),
        ("4. Consideraciones éticas y de comunicación",
         "Al presentar las gráficas hay que explicar qué datos usamos, cómo se ponderaron y qué "
         "valores faltan o se rellenaron. Debemos distinguir las respuestas de la encuesta de las "
         "denuncias y los totales de casos de los porcentajes. No sería correcto llamar violentos a "
         "ciertos estados ni asumir que en los que tienen menos casos el problema es menor. Si "
         "analizamos grupos más pequeños, tenemos que cuidar su privacidad. Los resultados son "
         "del archivo del curso y no deben presentarse como cifras oficiales de INEGI."),
    ]


def generar_pdf(r: dict) -> None:
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                  TableStyle, Image, PageBreak)
    from PIL import Image as PILImage
    RUTA_PDF.mkdir(parents=True, exist_ok=True)
    estilos = getSampleStyleSheet()
    estilos.add(ParagraphStyle("Texto", fontName="Helvetica", fontSize=9.3, leading=12.5,
                               spaceAfter=7, alignment=TA_LEFT))
    estilos.add(ParagraphStyle("Celda", fontName="Helvetica", fontSize=8, leading=10))
    estilos.add(ParagraphStyle("Nota", fontName="Helvetica", fontSize=8, leading=10.5, textColor=colors.HexColor("#475569"), spaceAfter=6))
    estilos["Title"].fontSize, estilos["Title"].leading = 21, 25
    estilos["Title"].textColor = colors.HexColor("#255F85")
    story = []
    def texto(s, estilo="Texto"):
        story.append(Paragraph(escape(s), estilos[estilo]))
    def titulo(s): texto(s, "Heading2")
    def figura(nombre, ancho, alto):
        with PILImage.open(RUTA_FIGURAS / nombre) as im: w, h = im.size
        escala = min(ancho/w, alto/h)
        story.append(Image(str(RUTA_FIGURAS / nombre), width=w*escala, height=h*escala))
        story.append(Spacer(1, 5))
    def nueva(): story.append(PageBreak())
    texto("ENDIREH 2021 | Práctica 4", "Title")
    texto("Medidas descriptivas, heterogeneidad y concentración", "Heading2")
    texto("Resumen ejecutivo del conjunto consolidado | Almacenes y Minería de Datos", "Nota")
    texto(f"Se analizaron {r['df'].height:,} registros. Los casos ponderados presentan concentración "
          f"territorial (Gini {r['concentracion']['gini']:.4f}), mientras que el estado civil muestra "
          "diversidad de categorías. Estos resultados descriptivos no identifican causas ni riesgos individuales.")
    titulo("Tabla de resultados")
    encabezados = ["Familia", "Variables", "Valor obtenido", "Interpretación"]
    datos = [[Paragraph(x, estilos["Celda"]) for x in encabezados]]
    datos += [[Paragraph(escape(v), estilos["Celda"]) for v in f.values()] for f in resumen(r)]
    tabla = Table(datos, colWidths=[68, 100, 155, 176], repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DBEAFE")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
    story.append(tabla)
    titulo("Alcance y decisiones")
    texto(f"La edad de primera unión tiene {100*r['df']['edad_primer_union'].null_count()/r['df'].height:.2f}% "
          "de faltantes. Los estadísticos de edad usan casos válidos. Hijos incluye ceros imputados. "
          "Media y mediana ponderadas usan factor_expansion. CV, IQR e índices de heterogeneidad son muestrales. "
          "Lorenz usa cantidades ponderadas, y la síntesis compara los dos índices sobre frecuencias ponderadas.")
    nueva()
    titulo("Localización y variabilidad")
    figura("02_distribucion_edad.png", 490, 210)
    texto("Figura 1. Histograma muestral de edad de primera unión. Solo casos válidos.", "Nota")
    figura("05_edad_por_violencia.png", 450, 230)
    texto("Figura 2. Boxplots muestrales por respuesta de violencia. Sin ponderar.", "Nota")
    loc = r["localizacion"].to_dicts()[0]
    texto(f"Edad: media ponderada {loc['media_ponderada']:.2f} años y mediana ponderada "
          f"{loc['mediana_ponderada']:.0f}. El IQR muestral es 13 años. Ambos grupos tienen mediana "
          "simple de 18 años e IQR de 13. Las diferencias descriptivas no demuestran causalidad. "
          "La abundancia de faltantes puede sesgar la selección de casos válidos.")
    texto("NumPy calcula la media y la mediana ponderadas (inversa empírica, sin interpolación). "
          "Polars calcula cuantiles simples y varianza. SciPy stats.variation calcula el CV muestral "
          "con ddof=1. Los notebooks separados contienen las tablas completas.")
    nueva()
    titulo("Heterogeneidad y concentración")
    figura("07_barras_iqv.png", 480, 220)
    texto("Figura 3. IQV muestral: se muestra una representación por variable, sin duplicar códigos y etiquetas.", "Nota")
    texto("Dinero propio y trabajo de la pareja presentan categorías casi equilibradas. Ahorros tiene menor "
          "diversidad. La asimetría de violencia describe respuestas de la muestra y no demuestra subregistro. "
          "Shannon está en bits, Gini-Simpson mide diversidad e IQV la normaliza según el número de categorías.")
    figura("08_curva_lorenz.png", 440, 220)
    texto("Figura 4. Lorenz de casos ponderados. Cada entidad es una unidad de igual peso en el eje horizontal.", "Nota")
    texto(f"Gini = {r['concentracion']['gini']:.4f}, área = {r['concentracion']['area_lorenz']:.4f}. "
          "El cálculo por frecuencias ordenadas coincide con 1 - 2 × área. La curva refleja desigualdad "
          "de cantidades y también diferencias de población. No permite asignar un riesgo a cada entidad.")
    nueva()
    titulo("Porcentaje ponderado por entidad")
    figura("06_violencia_por_entidad.png", 465, 535)
    texto("Figura 5. Porcentaje de respuesta afirmativa entre la población representada por cada entidad: "
          "100 × suma(violencia × factor_expansion) / suma(factor_expansion).", "Nota")
    texto("Esta figura incorpora el denominador poblacional, a diferencia de Lorenz. Es una prevalencia "
          "descriptiva del indicador disponible en el archivo del curso. No se presentan intervalos porque "
          "no se dispone de todas las variables del diseño muestral. No debe confundirse con denuncia "
          "administrativa ni presentarse como estimación oficial de INEGI.")
    nueva()
    titulo("Síntesis: Gini y entropía sobre estado civil")
    texto(sintesis_texto(r))
    titulo("Cuestionario y reflexión")
    for pregunta, respuesta in cuestionario(r):
        texto(pregunta, "Heading3")
        texto(respuesta)
    nueva()
    titulo("Método, reproducibilidad y fuentes")
    texto(metodologia(r))
    titulo("Cómo reproducir la entrega")
    texto("Desde la raíz: activar .venv, instalar requirements.txt y ejecutar python -m src.cleaning.limpiar_datos "
          "si no existe el archivo procesado. Después, python -m src.reporting.practica4 recalcula "
          "figuras y tablas y genera este PDF. Los notebooks presentan cada análisis por separado. "
          "Los datos originales y procesados no se versionan. El CSV original se entrega por separado.")
    titulo("Fuentes")
    texto("INEGI (2022). ENDIREH 2021. Fuente del consolidado utilizado: "
          "https://www.inegi.org.mx/programas/endireh/2021/", "Nota")
    texto("INEGI. ENDIREH 2021, estructura de la base de datos. Referencia para revisar "
          "códigos y universos. La revisión del consolidado sigue siendo una mejora pendiente.", "Nota")
    texto("Gini, C. (1912). Variabilità e mutabilità. Shannon, C. E. (1948). A Mathematical Theory "
          "of Communication. Wilcox, A. R. (1973). Indices of Qualitative Variation and Political Measurement.", "Nota")
    texto("AyMD_Practica_4.pdf, material del curso. Las referencias a pasos 8 y 9 se interpretan "
          "como las visualizaciones y la tabla resumen del desarrollo. Se conserva la arquitectura "
          "de la práctica 3 y la ausencia de edades se declara explícitamente.", "Nota")
    def pie(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(48, 38, A4[0]-48, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(48, 25, "ENDIREH 2021 | Consolidado del proyecto | Práctica 4")
        canvas.drawRightString(A4[0]-48, 25, str(doc.page))
    doc = SimpleDocTemplate(str(ARCHIVO_REPORTE_P4), pagesize=A4, rightMargin=48,
                            leftMargin=48, topMargin=42, bottomMargin=50,
                            title="ENDIREH 2021 - Reporte de práctica 4", author="Proyecto ENDIREH")
    doc.build(story, onFirstPage=pie, onLaterPages=pie)


def main() -> None:
    generar_eda()
    r = resultados()
    generar_grafica_iqv(r["heterogeneidad"])
    generar_pdf(r)
    print(f"Reporte: {ARCHIVO_REPORTE_P4}")


if __name__ == "__main__":
    main()
