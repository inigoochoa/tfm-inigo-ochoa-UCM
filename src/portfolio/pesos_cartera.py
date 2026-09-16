"""Cartera long-only equal-weight entre los 7 factores, con un tilt acotado hacia Momentum proporcional al z-score de su predicción de Capa 2 (horizonte largo)."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS

FACTORES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "bab_usa"]
PESO_BASE = 1 / len(FACTORES)  # ≈ 0.142857

Z_CAP = 2.0
TILT_MAX = 0.07
PESO_MIN = 0.01

NOMBRE_FICHERO_PREDICCIONES_LARGO = "predicciones_test_largo.csv"
NOMBRE_FICHERO_PESOS_SALIDA = "pesos_cartera.csv"


def calcular_zscore_expanding(pred_momentum: pd.Series) -> pd.Series:
    # shift(1): el z-score de cada mes usa solo media/std hasta el mes anterior (anti-leakage).
    # NaN en los primeros meses se resuelve en calcular_tilt() como tilt=0.
    media_expanding = pred_momentum.expanding().mean().shift(1)
    std_expanding = pred_momentum.expanding().std().shift(1)
    return (pred_momentum - media_expanding) / std_expanding


def calcular_tilt(z: pd.Series) -> pd.Series:
    z_capado = z.clip(lower=-Z_CAP, upper=Z_CAP)
    tilt = TILT_MAX * (z_capado / Z_CAP)
    return tilt.fillna(0.0)


def calcular_pesos_cartera(pred_momentum_largo: pd.Series) -> pd.DataFrame:
    z = calcular_zscore_expanding(pred_momentum_largo)
    tilt = calcular_tilt(z)

    pesos = pd.DataFrame(PESO_BASE, index=pred_momentum_largo.index, columns=FACTORES)
    pesos["Mom"] = PESO_BASE + tilt

    otros = [f for f in FACTORES if f != "Mom"]
    ajuste_por_factor = -tilt / len(otros)
    for f in otros:
        pesos[f] = PESO_BASE + ajuste_por_factor

    pesos = pesos.clip(lower=PESO_MIN)
    pesos = pesos.div(pesos.sum(axis=1), axis=0)  # renormalizar tras el suelo

    return pesos


if __name__ == "__main__":
    predicciones_largo = pd.read_csv(
        f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PREDICCIONES_LARGO}",
        parse_dates=["fecha_mes"],
        index_col="fecha_mes",
    )
    pred_momentum_largo = predicciones_largo["Mom"]

    pesos = calcular_pesos_cartera(pred_momentum_largo)

    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PESOS_SALIDA}"
    pesos.to_csv(ruta_salida)

    print(f"Pesos de cartera guardados en: {ruta_salida}")
    print("\nEstadísticos de los pesos (verificación rápida):")
    print(pesos.describe().round(4))
    print(
        f"\nSuma de pesos por fila (debe ser 1.0 siempre): "
        f"min={pesos.sum(axis=1).min():.6f}, max={pesos.sum(axis=1).max():.6f}"
    )
    print(
        f"\nRango del peso de Momentum: "
        f"min={pesos['Mom'].min():.4f}, max={pesos['Mom'].max():.4f} "
        f"(base sin tilt: {PESO_BASE:.4f})"
    )