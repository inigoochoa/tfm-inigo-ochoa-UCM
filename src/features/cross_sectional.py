"""Relative strength (momentum del propio factor, t-12 a t-1) como feature adicional del Random Forest de Capa 2."""

import numpy as np
import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS

VENTANA_RELATIVE_STRENGTH_MESES = 12
MESES_EXCLUIDOS_RECIENTES = 1  # excluye t-0, ventana efectiva t-12 a t-1

FACTORES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "bab_usa"]


def cargar_panel_features_macro() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_features_macro.csv"
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)


def calcular_relative_strength(
    serie_factor: pd.Series,
    ventana: int = VENTANA_RELATIVE_STRENGTH_MESES,
    meses_excluidos: int = MESES_EXCLUIDOS_RECIENTES,
) -> pd.Series:
    """Retorno acumulado compuesto de `ventana` meses, desplazado `meses_excluidos` meses para excluir los más recientes (t-12 a t-1 por defecto)."""
    # log1p -> rolling().sum() -> expm1: retorno compuesto vectorizado, equivalente a
    # rolling().apply(prod) pero sin sus problemas de rendimiento/NaN en el borde.
    log_retornos = np.log1p(serie_factor)
    log_acumulado = log_retornos.rolling(window=ventana, min_periods=ventana).sum()
    retorno_acumulado = np.expm1(log_acumulado)

    relative_strength = retorno_acumulado.shift(meses_excluidos)

    return relative_strength


def construir_relative_strength(factores: list[str] = FACTORES) -> pd.DataFrame:
    panel = cargar_panel_features_macro()

    df_rs = pd.DataFrame(index=panel.index)
    for factor in factores:
        if factor not in panel.columns:
            print(f"Aviso: factor '{factor}' no encontrado en el panel, se omite.")
            continue
        df_rs[f"rs_{factor}"] = calcular_relative_strength(panel[factor])

    print(f"Relative strength construido para {len(df_rs.columns)} factores.")
    print(f"Shape: {df_rs.shape}")

    return df_rs


def guardar_relative_strength() -> None:
    panel = cargar_panel_features_macro()
    df_rs = construir_relative_strength()

    panel_completo = panel.join(df_rs, how="left")

    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_features_completo.csv"
    panel_completo.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}")
    print(f"Shape final: {panel_completo.shape}")


if __name__ == "__main__":
    guardar_relative_strength()