"""Orquesta las señales cross-sectional de Capa 2 sobre fundamentales_pit.parquet: sort variables, valuation composite, dispersion y value spread, winsorizados, en dos paneles (ticker-mes y mercado-mes)."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from features.senales_cross_sectional import (
    cargar_fundamentales_pit,
    calcular_sort_variable_smb,
    calcular_sort_variable_hml,
    calcular_sort_variable_rmw,
    calcular_sort_variable_cma,
    calcular_valuation_composite,
    calcular_dispersion,
    calcular_valuation_composite_medio,
    calcular_value_spread,
)

NOMBRE_FICHERO_PANEL_TICKER = "panel_senales_ticker_capa2.parquet"
NOMBRE_FICHERO_PANEL_MERCADO = "panel_senales_mercado_capa2.csv"

COLUMNAS_SORT_POR_FACTOR = {
    "SMB": "sort_smb",
    "HML": "sort_hml",
    "RMW": "sort_rmw",
    "CMA": "sort_cma",
}

COLUMNAS_NECESARIAS_VALUATION = {"netinc", "marketcap", "ncfo", "opinc"}

# Percentiles de winsorización por columna, fijados tras validación empírica de
# outliers extremos (fusiones, recompras agresivas, ratios con denominador ~0).
# sort_smb queda fuera a propósito: su cola alta es señal genuina de tamaño (megacaps).
WINSORIZACION_POR_COLUMNA = {
    "sort_hml": (0.005, 0.995),
    "sort_rmw": (0.01, 0.99),
    "sort_cma": (0.01, 0.99),
    "earnings_yield": (0.001, 0.999),
    "cash_flow_yield": (0.005, 0.995),
}


def winsorizar_por_mes(
    df: pd.DataFrame,
    columna: str,
    percentil_inferior: float,
    percentil_superior: float,
) -> pd.Series:
    # Por fecha_mes (no sobre la serie completa): un outlier de un mes no debe
    # distorsionar los cortes de leg largo/corto de otros meses.
    inferior = df.groupby("fecha_mes")[columna].transform(
        lambda s: s.quantile(percentil_inferior)
    )
    superior = df.groupby("fecha_mes")[columna].transform(
        lambda s: s.quantile(percentil_superior)
    )
    return df[columna].clip(lower=inferior, upper=superior)


def construir_senales_capa2(
    fundamentales_pit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    faltantes = COLUMNAS_NECESARIAS_VALUATION - set(fundamentales_pit.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias para valuation composite: {faltantes}"
        )

    df = fundamentales_pit.copy()

    df["sort_smb"] = calcular_sort_variable_smb(df)
    df["sort_hml"] = calcular_sort_variable_hml(df)
    df["sort_rmw"] = calcular_sort_variable_rmw(df)
    df["sort_cma"] = calcular_sort_variable_cma(df)

    for columna, (p_inf, p_sup) in WINSORIZACION_POR_COLUMNA.items():
        if columna in df.columns:
            df[columna] = winsorizar_por_mes(df, columna, p_inf, p_sup)

    df = calcular_valuation_composite(df)
    for columna in ("earnings_yield", "cash_flow_yield"):
        p_inf, p_sup = WINSORIZACION_POR_COLUMNA[columna]
        df[columna] = winsorizar_por_mes(df, columna, p_inf, p_sup)

    # Recalcula el composite (mismo z-score que calcular_valuation_composite) ahora
    # que earnings_yield/cash_flow_yield están winsorizados, para que el z-score
    # cross-sectional no quede contaminado por el ratio crudo.
    def _zscore_por_fecha(serie: pd.Series) -> pd.Series:
        media = serie.groupby(df["fecha_mes"]).transform("mean")
        desviacion = serie.groupby(df["fecha_mes"]).transform("std")
        return (serie - media) / desviacion.replace(0, pd.NA)

    df["z_earnings_yield"] = _zscore_por_fecha(df["earnings_yield"])
    df["z_cash_flow_yield"] = _zscore_por_fecha(df["cash_flow_yield"])
    df["valuation_composite"] = df[["z_earnings_yield", "z_cash_flow_yield"]].mean(
        axis=1, skipna=True
    )

    panel_ticker = df[
        [
            "ticker",
            "fecha_mes",
            "sort_smb",
            "sort_hml",
            "sort_rmw",
            "sort_cma",
            "earnings_yield",
            "cash_flow_yield",
            "valuation_composite",
        ]
    ].copy()

    panel_mercado = calcular_dispersion(df)
    panel_mercado = panel_mercado.join(calcular_valuation_composite_medio(df), how="outer")

    spreads_por_factor = calcular_value_spread(df, COLUMNAS_SORT_POR_FACTOR)
    for factor, spread_df in spreads_por_factor.items():
        spread_renombrado = spread_df.rename(
            columns={
                "value_spread": f"value_spread_{factor}",
                "z_value_spread": f"z_value_spread_{factor}",
            }
        )
        panel_mercado = panel_mercado.join(spread_renombrado, how="outer")

    panel_mercado = panel_mercado.sort_index().reset_index()

    print(
        f"Señales Capa 2 construidas: panel_ticker "
        f"{panel_ticker.shape[0]} filas x {panel_ticker['ticker'].nunique()} "
        f"tickers; panel_mercado {panel_mercado.shape[0]} meses."
    )

    return panel_ticker, panel_mercado


def guardar_senales_capa2(
    panel_ticker: pd.DataFrame,
    panel_mercado: pd.DataFrame,
    ruta_directorio: str = RUTA_DATOS_PROCESADOS,
    nombre_fichero_ticker: str = NOMBRE_FICHERO_PANEL_TICKER,
    nombre_fichero_mercado: str = NOMBRE_FICHERO_PANEL_MERCADO,
) -> None:
    ruta_ticker = f"{ruta_directorio}/{nombre_fichero_ticker}"
    ruta_mercado = f"{ruta_directorio}/{nombre_fichero_mercado}"

    panel_ticker.to_parquet(ruta_ticker, index=False)
    panel_mercado.to_csv(ruta_mercado, index=False)

    print(f"Guardado panel_ticker en {ruta_ticker} — shape {panel_ticker.shape}")
    print(f"Guardado panel_mercado en {ruta_mercado} — shape {panel_mercado.shape}")


if __name__ == "__main__":
    fundamentales_pit = cargar_fundamentales_pit()
    panel_ticker, panel_mercado = construir_senales_capa2(fundamentales_pit)

    print("\nEjemplo panel_ticker (AAPL, últimos 5 meses):")
    print(
        panel_ticker[panel_ticker["ticker"] == "AAPL"]
        .tail(5)
        .to_string(index=False)
    )

    print("\nEjemplo panel_mercado (últimos 5 meses):")
    print(panel_mercado.tail(5).to_string(index=False))

    print("\nVerificación post-winsorización (máximos por columna):")
    print(
        panel_ticker[
            ["sort_smb", "sort_hml", "sort_rmw", "sort_cma",
             "earnings_yield", "cash_flow_yield", "valuation_composite"]
        ].max()
    )

    guardar_senales_capa2(panel_ticker, panel_mercado)