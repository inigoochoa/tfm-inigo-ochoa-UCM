"""Features macro derivadas para el modelo de régimen: spread de tipos, retorno y drawdown móvil del S&P500, volatilidad realizada, y z-scores móviles de 60m para variables con deriva secular."""

import numpy as np
import pandas as pd

from config.series import RUTA_DATOS_CRUDOS, RUTA_DATOS_PROCESADOS

VENTANA_DRAWDOWN_MESES = 12
DIAS_TRADING_ANUAL = 252

VENTANA_ZSCORE_MESES = 60

COLUMNAS_ZSCORE_MOVIL = [
    "tipo_10y",
    "tipo_2y",
    "fed_funds",
    "tasa_paro",
    "tasa_ahorro",
]


def cargar_panel_mensual() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_mensual.csv"
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)


def cargar_mercado_yahoo_diario() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_CRUDOS}/mercado_yahoo.csv"
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)


def calcular_spread_tipos(panel: pd.DataFrame) -> pd.Series:
    return panel["tipo_10y"] - panel["tipo_2y"]


def calcular_retorno_mensual_sp500(panel: pd.DataFrame) -> pd.Series:
    return panel["sp500"].pct_change(fill_method=None)


def calcular_drawdown_movil_sp500(
    panel: pd.DataFrame, ventana_meses: int = VENTANA_DRAWDOWN_MESES
) -> pd.Series:
    # Máximo de una ventana móvil (no el máximo histórico absoluto), que arrastraría
    # picos de hace décadas y sería menos útil para detección de régimen.
    maximo_movil = panel["sp500"].rolling(window=ventana_meses, min_periods=ventana_meses).max()
    drawdown = panel["sp500"] / maximo_movil - 1
    return drawdown


def calcular_volatilidad_realizada_mensual(
    df_diario: pd.DataFrame,
) -> pd.Series:
    # Anualizada (x sqrt(252)) para ser comparable con el VIX.
    retornos_diarios = df_diario["sp500"].pct_change(fill_method=None)
    vol_mensual = retornos_diarios.resample("ME").std() * np.sqrt(DIAS_TRADING_ANUAL)
    vol_mensual.name = "vol_realizada_sp500"
    return vol_mensual


def calcular_zscore_movil(
    serie: pd.Series, ventana_meses: int = VENTANA_ZSCORE_MESES
) -> pd.Series:
    # Z-score respecto a media/std móviles (no absolutas): evita que el clustering de
    # régimen confunda deriva secular (p. ej. tipos estructuralmente más bajos
    # post-2008) con dinámica cíclica genuina.
    media_movil = serie.rolling(window=ventana_meses, min_periods=ventana_meses).mean()
    std_movil = serie.rolling(window=ventana_meses, min_periods=ventana_meses).std()
    return (serie - media_movil) / std_movil


def anadir_zscores_moviles(panel: pd.DataFrame) -> pd.DataFrame:
    for columna in COLUMNAS_ZSCORE_MOVIL:
        panel[f"{columna}_zscore_{VENTANA_ZSCORE_MESES}m"] = calcular_zscore_movil(panel[columna])

    print(f"Z-scores móviles ({VENTANA_ZSCORE_MESES}m) añadidos para: {COLUMNAS_ZSCORE_MOVIL}")

    return panel


def construir_features_macro() -> pd.DataFrame:
    panel = cargar_panel_mensual()
    df_diario = cargar_mercado_yahoo_diario()

    panel["spread_10y_2y"] = calcular_spread_tipos(panel)
    panel["retorno_sp500"] = calcular_retorno_mensual_sp500(panel)
    panel["drawdown_sp500_12m"] = calcular_drawdown_movil_sp500(panel)

    vol_realizada = calcular_volatilidad_realizada_mensual(df_diario)
    panel = panel.join(vol_realizada, how="left")

    panel = anadir_zscores_moviles(panel)

    print(f"Features macro construidas. Shape final: {panel.shape}")
    print(f"Nuevas columnas: spread_10y_2y, retorno_sp500, "
          f"drawdown_sp500_12m, vol_realizada_sp500, "
          f"{len(COLUMNAS_ZSCORE_MOVIL)} columnas zscore_60m")

    return panel


def guardar_features_macro() -> None:
    panel = construir_features_macro()
    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_features_macro.csv"
    panel.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}")


if __name__ == "__main__":
    guardar_features_macro()