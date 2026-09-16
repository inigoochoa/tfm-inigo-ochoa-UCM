"""Compara el rendimiento de la cartera con tilt de Momentum (mensual y con rebalanceo anual) frente a equal-weight puro, sobre el periodo de test."""

import numpy as np
import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS

FACTORES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "bab_usa"]
PESO_BASE = 1 / len(FACTORES)

MESES_POR_ANIO = 12

NOMBRE_FICHERO_PESOS = "pesos_cartera.csv"
NOMBRE_FICHERO_PANEL_MAESTRO = "panel_maestro_capa2.csv"
NOMBRE_FICHERO_SALIDA = "comparacion_carteras.csv"


def cargar_pesos_y_retornos():
    pesos = pd.read_csv(
        f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PESOS}",
        parse_dates=["fecha_mes"],
        index_col="fecha_mes",
    )

    panel = pd.read_csv(
        f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PANEL_MAESTRO}",
        parse_dates=["fecha_mes"],
    ).set_index("fecha_mes")

    retornos = panel[FACTORES].loc[pesos.index]

    return pesos, retornos


def calcular_pesos_rebalanceo_anual(pesos_mensuales: pd.DataFrame) -> pd.DataFrame:
    """Fija el peso del primer mes de cada bloque de 12 y lo mantiene constante (forward-fill) durante los 12 meses siguientes."""
    fechas_rebalanceo = pesos_mensuales.index[::MESES_POR_ANIO]
    pesos_en_fechas_rebalanceo = pesos_mensuales.loc[fechas_rebalanceo]

    pesos_anuales = pesos_en_fechas_rebalanceo.reindex(
        pesos_mensuales.index, method="ffill"
    )

    return pesos_anuales


def calcular_retorno_cartera(pesos: pd.DataFrame, retornos: pd.DataFrame) -> pd.Series:
    return (pesos * retornos).sum(axis=1)


def calcular_metricas(retorno_mensual: pd.Series, nombre: str) -> dict:
    # Sharpe sin restar tasa libre de riesgo: los factores ya están en exceso de RF
    # (salvo bab_usa, long-short, que no la necesita).
    media_mensual = retorno_mensual.mean()
    std_mensual = retorno_mensual.std()
    sharpe_anualizado = (media_mensual / std_mensual) * np.sqrt(MESES_POR_ANIO)

    riqueza_acumulada = (1 + retorno_mensual).cumprod()
    pico_acumulado = riqueza_acumulada.cummax()
    drawdown = (riqueza_acumulada / pico_acumulado) - 1
    max_drawdown = drawdown.min()

    n_meses = len(retorno_mensual)
    capital_final = riqueza_acumulada.iloc[-1]
    # CAGR real, no media aritmética mensual compuesta (esta última sobreestima
    # el crecimiento por la desigualdad de Jensen).
    retorno_anualizado = capital_final ** (MESES_POR_ANIO / n_meses) - 1
    calmar = retorno_anualizado / abs(max_drawdown)

    return {
        "cartera": nombre,
        "retorno_anualizado": retorno_anualizado,
        "volatilidad_anualizada": std_mensual * np.sqrt(MESES_POR_ANIO),
        "sharpe": sharpe_anualizado,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


if __name__ == "__main__":
    pesos_tilt, retornos = cargar_pesos_y_retornos()

    pesos_equal_weight = pd.DataFrame(
        PESO_BASE, index=pesos_tilt.index, columns=FACTORES
    )
    pesos_tilt_anual = calcular_pesos_rebalanceo_anual(pesos_tilt)

    retorno_tilt = calcular_retorno_cartera(pesos_tilt, retornos)
    retorno_equal_weight = calcular_retorno_cartera(pesos_equal_weight, retornos)
    retorno_tilt_anual = calcular_retorno_cartera(pesos_tilt_anual, retornos)

    resultados = pd.DataFrame([
        calcular_metricas(retorno_tilt, "Con tilt (mensual)"),
        calcular_metricas(retorno_tilt_anual, "Con tilt (rebalanceo anual)"),
        calcular_metricas(retorno_equal_weight, "Equal-weight puro"),
    ]).set_index("cartera")

    print("=== Comparación de rendimiento: mensual vs. rebalanceo anual vs. equal-weight ===\n")
    print(resultados.round(4).to_string())

    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_SALIDA}"
    resultados.to_csv(ruta_salida)
    print(f"\nResultados guardados en: {ruta_salida}")