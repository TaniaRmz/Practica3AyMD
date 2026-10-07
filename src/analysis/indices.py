"""Funciones para calcular índices y cuantiles ponderados."""

import numpy as np


def frecuencias_validas(frecuencias) -> np.ndarray:
    valores = np.asarray(frecuencias, dtype=float)
    if valores.ndim != 1 or valores.size == 0:
        raise ValueError("Se requiere un vector no vacío de frecuencias.")
    if not np.isfinite(valores).all() or (valores < 0).any() or valores.sum() <= 0:
        raise ValueError("Las frecuencias deben ser finitas, no negativas y sumar más de cero.")
    return valores


def indices_heterogeneidad(frecuencias) -> dict[str, float | int]:
    """Calcula Gini-Simpson, IQV y Shannon con las frecuencias de cada categoría."""
    valores = frecuencias_validas(frecuencias)
    valores = valores[valores > 0]
    k = valores.size
    p = valores / valores.sum()
    gs = float(1 - np.sum(p ** 2))
    return {
        "k_categorias": int(k),
        "Gini_Simpson": gs,
        "IQV": float(k / (k - 1) * gs) if k > 1 else 0.0,
        "Shannon_Entropy": float(-np.sum(p * np.log2(p))),
    }


def curva_lorenz(cantidades) -> tuple[np.ndarray, np.ndarray]:
    """Cada posición representa una unidad con el mismo peso (p. ej. una entidad)."""
    valores = np.sort(frecuencias_validas(cantidades))
    return np.linspace(0, 1, valores.size + 1), np.r_[0.0, np.cumsum(valores) / valores.sum()]


def coeficiente_gini(cantidades) -> float:
    """Gini no corregido: máximo (n-1)/n para n unidades, no exactamente 1."""
    valores = np.sort(frecuencias_validas(cantidades))
    n = valores.size
    return float(2 * np.dot(np.arange(1, n + 1), valores) / (n * valores.sum()) - (n + 1) / n)


def cuantil_ponderado(valores, pesos, q: float = 0.5) -> float:
    """Inversa de la distribución empírica ponderada, sin interpolación.

    Devuelve el primer valor cuya proporción acumulada de peso alcanza q.
    Elimina pesos cero y exige pares completos, finitos y pesos no negativos.
    """
    x, w = np.asarray(valores, dtype=float), np.asarray(pesos, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or not 0 <= q <= 1:
        raise ValueError("Valores/pesos incompatibles o cuantil fuera de [0, 1].")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Se requieren pares finitos y pesos no negativos.")
    seleccion = w > 0
    if not seleccion.any():
        raise ValueError("No hay peso positivo.")
    x, w = x[seleccion], w[seleccion]
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    posicion = np.searchsorted(np.cumsum(w), q * w.sum(), side="left")
    return float(x[min(posicion, x.size - 1)])
