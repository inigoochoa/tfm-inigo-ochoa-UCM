"""Señales cross-sectional de Capa 2 sobre fundamentales_pit.parquet: sort variables por factor, valuation composite, dispersion y value spread. Orquestado desde pipeline_senales_capa2.py."""

from pathlib import Path

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS

NOMBRE_FICHERO_FUNDAMENTALES_PIT = "fundamentales_pit.parquet"


def cargar_fundamentales_pit(
    ruta: Path = Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_FUNDAMENTALES_PIT,
) -> pd.DataFrame:
    df = pd.read_parquet(ruta)

    columnas_necesarias = {
        "ticker", "fecha_mes", "marketcap", "equity", "ebit", "intexp",
        "assets", "ppnenet", "depamor", "inventory",
    }
    faltantes = columnas_necesarias - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias en fundamentales_pit: {faltantes}"
        )

    return df


def calcular_sort_variable_smb(df: pd.DataFrame) -> pd.Series:
    return df["marketcap"]


def calcular_sort_variable_hml(df: pd.DataFrame) -> pd.Series:
    # book-to-market = equity / marketcap; equity negativo o cero -> NaN (pierde
    # sentido económico, convención estándar Fama-French).
    equity = df["equity"].where(df["equity"] > 0)
    marketcap = df["marketcap"].replace(0, pd.NA)
    return equity / marketcap


def calcular_sort_variable_rmw(df: pd.DataFrame) -> pd.Series:
    # rentabilidad operativa = (ebit - intexp) / equity; mismo criterio que HML
    # para equity negativo o cero -> NaN.
    ebit = df["ebit"]
    intexp = df["intexp"]
    equity = df["equity"].where(df["equity"] > 0)
    return (ebit - intexp) / equity


def calcular_sort_variable_cma(df: pd.DataFrame) -> pd.Series:
    # Aproximación al I/A del q-factor model: (Δ ppnenet + depamor + Δ inventory) /
    # assets_(t-1). inventory ausente se trata como 0 (sectores sin inventario);
    # el resto de ausencias o assets_(t-1) <= 0 -> NaN.
    orden_original = df.index
    df_ordenado = df.sort_values(["ticker", "fecha_mes"])

    inventory_rellenada = df_ordenado["inventory"].fillna(0)
    grupo_ticker = df_ordenado["ticker"]

    ppnenet_prev = df_ordenado.groupby("ticker")["ppnenet"].shift(1)
    inventory_prev = inventory_rellenada.groupby(grupo_ticker).shift(1)
    assets_prev = df_ordenado.groupby("ticker")["assets"].shift(1)

    delta_ppnenet = df_ordenado["ppnenet"] - ppnenet_prev
    delta_inventory = inventory_rellenada - inventory_prev

    numerador = delta_ppnenet + df_ordenado["depamor"] + delta_inventory
    denominador = assets_prev.where(assets_prev > 0)

    resultado = numerador / denominador
    return resultado.reindex(orden_original)


def calcular_valuation_composite(df: pd.DataFrame) -> pd.DataFrame:
    # earnings_yield = netinc/marketcap; cash_flow_yield = ncfo/opinc (opinc <= 0 ->
    # NaN). Composite = media de los z-scores cross-sectional no nulos: si falta uno,
    # se usa solo el otro en vez de excluir la fila.
    df = df.copy()

    df["earnings_yield"] = df["netinc"] / df["marketcap"].replace(0, pd.NA)

    opinc_valido = df["opinc"].where(df["opinc"] > 0)
    df["cash_flow_yield"] = df["ncfo"] / opinc_valido

    def _zscore_por_fecha(serie: pd.Series) -> pd.Series:
        media = serie.groupby(df["fecha_mes"]).transform("mean")
        desviacion = serie.groupby(df["fecha_mes"]).transform("std")
        return (serie - media) / desviacion.replace(0, pd.NA)

    df["z_earnings_yield"] = _zscore_por_fecha(df["earnings_yield"])
    df["z_cash_flow_yield"] = _zscore_por_fecha(df["cash_flow_yield"])

    df["valuation_composite"] = df[["z_earnings_yield", "z_cash_flow_yield"]].mean(
        axis=1, skipna=True
    )

    return df

def calcular_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    # Desviación estándar cross-sectional del valuation_composite por fecha_mes:
    # una única serie de mercado (no por factor ni a nivel de acción).
    dispersion = (
        df.groupby("fecha_mes")["valuation_composite"]
        .std()
        .rename("dispersion")
        .to_frame()
    )
    return dispersion

def calcular_valuation_composite_medio(df: pd.DataFrame) -> pd.DataFrame:
    # Media cross-sectional del valuation_composite por fecha_mes: nivel agregado de
    # valoración del mercado (complementa a dispersion, que mide su forma, no su nivel).
    media = (
        df.groupby("fecha_mes")["valuation_composite"]
        .mean()
        .rename("valuation_composite_medio")
        .to_frame()
    )
    return media

QUANTIL_LEGS_POR_DEFECTO = 0.2  # quintiles

# Convención académica fija (Fama-French / q-factor model), no configurable:
# True -> largo = sort variable alto (HML, RMW); False -> largo = sort variable bajo.
_SENTIDO_LARGO_ASCENDENTE = {
    "SMB": False,
    "HML": True,
    "RMW": True,
    "CMA": False,
}


def _asignar_legs(
    df: pd.DataFrame,
    columna_sort: str,
    ascendente: bool,
    quantil: float = QUANTIL_LEGS_POR_DEFECTO,
) -> pd.Series:
    # Asigna leg 'largo'/'corto'/NaN (zona neutra) según percentiles [quantil, 1-quantil]
    # del sort variable, cross-sectional por fecha_mes.
    corte_inferior = df.groupby("fecha_mes")[columna_sort].transform(
        lambda s: s.quantile(quantil)
    )
    corte_superior = df.groupby("fecha_mes")[columna_sort].transform(
        lambda s: s.quantile(1 - quantil)
    )

    es_alto = df[columna_sort] >= corte_superior
    es_bajo = df[columna_sort] <= corte_inferior

    legs = pd.Series(pd.NA, index=df.index, dtype="object")
    if ascendente:
        legs = legs.mask(es_alto, "largo").mask(es_bajo, "corto")
    else:
        legs = legs.mask(es_bajo, "largo").mask(es_alto, "corto")

    return legs


def calcular_value_spread(
    df: pd.DataFrame,
    columnas_sort_por_factor: dict[str, str],
    quantil: float = QUANTIL_LEGS_POR_DEFECTO,
) -> dict[str, pd.DataFrame]:
    # Value spread por factor: valuation_composite medio del leg largo menos el del
    # leg corto, y su z-score respecto a media/std expanding (no ventana de 60m: la
    # cobertura real ya es corta, 1998-2026). shift(1) en el z-score: anti-leakage,
    # el mes t usa solo spreads observados hasta t-1.
    factores_desconocidos = set(columnas_sort_por_factor) - set(_SENTIDO_LARGO_ASCENDENTE)
    if factores_desconocidos:
        raise ValueError(
            f"Factores sin convención de sentido definida: {factores_desconocidos}"
        )

    resultados = {}

    for factor, columna_sort in columnas_sort_por_factor.items():
        ascendente = _SENTIDO_LARGO_ASCENDENTE[factor]
        legs = _asignar_legs(df, columna_sort, ascendente, quantil)

        valoracion_por_leg = (
            df.assign(_leg=legs)
            .groupby(["fecha_mes", "_leg"])["valuation_composite"]
            .mean()
            .unstack("_leg")
        )

        spread = valoracion_por_leg["largo"] - valoracion_por_leg["corto"]
        media_expanding = spread.expanding().mean().shift(1)
        std_expanding = spread.expanding().std().shift(1)
        z_spread = (spread - media_expanding) / std_expanding.replace(0, pd.NA)

        resultados[factor] = pd.DataFrame(
            {"value_spread": spread, "z_value_spread": z_spread}
        )

    return resultados