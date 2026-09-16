"""Descarga de Yahoo Finance: S&P500/VIX agregados y precios individuales de tickers activos (complemento a los delistados de Sharadar)."""

import time

import pandas as pd
import yfinance as yf

from config.series import TICKERS_YAHOO, FECHA_INICIO, FECHA_FIN, RUTA_DATOS_CRUDOS
from data_ingestion.universe import tickers_por_estado

TAMANO_LOTE_YAHOO = 50
PAUSA_ENTRE_LOTES_SEGUNDOS = 2

# Mapeo de tickers cuya notación difiere entre Sharadar y Yahoo
# Finance (clases de acciones múltiples: Sharadar usa punto, Yahoo
# usa guion).
MAPEO_TICKERS_SHARADAR_A_YAHOO = {
    "BF.B": "BF-B",
    "BRK.B": "BRK-B",
}


def _descargar_ticker(ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.Series:
    datos = yf.download(
        ticker,
        start=fecha_inicio,
        end=fecha_fin,
        progress=False,
        auto_adjust=False
    )
    serie = datos["Close"]
    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:,0]
    serie.name = ticker
    return serie

def descargar_todas_series_yahoo(
        fecha_inicio: str = FECHA_INICIO,
        fecha_fin: str = FECHA_FIN,
        ruta_salida: str = f"{RUTA_DATOS_CRUDOS}/mercado_yahoo.csv",
) -> pd.DataFrame:
    series = []
    for ticker, nombre in TICKERS_YAHOO.items():
        print(f"Descargando {nombre} ({ticker})...")
        serie = _descargar_ticker(ticker, fecha_inicio, fecha_fin)
        serie.name = nombre
        series.append(serie)

    df_combinado = pd.concat(series, axis=1)
    df_combinado.index.name = "fecha"

    df_combinado.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}. Shape final: {df_combinado.shape}")

    return df_combinado


def identificar_tickers_activos() -> list[str]:
    tickers_activos = tickers_por_estado("N")

    tickers_activos_yahoo = {
        MAPEO_TICKERS_SHARADAR_A_YAHOO.get(t, t) for t in tickers_activos
    }

    print(f"Tickers activos a descargar vía Yahoo: {len(tickers_activos_yahoo)}")
    return sorted(tickers_activos_yahoo)


def convertir_ticker_yahoo_a_sharadar(ticker_yahoo: str) -> str:
    mapeo_inverso = {v: k for k, v in MAPEO_TICKERS_SHARADAR_A_YAHOO.items()}
    return mapeo_inverso.get(ticker_yahoo, ticker_yahoo)


def descargar_precios_individuales_yahoo(tickers: list[str]) -> pd.DataFrame:
    lotes = [
        tickers[i : i + TAMANO_LOTE_YAHOO]
        for i in range(0, len(tickers), TAMANO_LOTE_YAHOO)
    ]

    dataframes_lotes = []
    for i, lote in enumerate(lotes):
        print(f"Descargando lote {i + 1}/{len(lotes)} ({len(lote)} tickers)...")
        datos = yf.download(
            tickers=lote,
            start=FECHA_INICIO,
            auto_adjust=True,  # 'Close' ya viene ajustado por splits/dividendos
            progress=False,
            group_by="ticker",
        )

        # Con múltiples tickers, yf.download devuelve columnas
        # MultiIndex (ticker, campo). Extraemos solo 'Close' (ya
        # ajustado) y pasamos a formato largo.
        if len(lote) == 1:
            # Con un solo ticker, no hay MultiIndex; se normaliza igual.
            df_close = datos[["Close"]].copy()
            df_close.columns = [lote[0]]
        else:
            df_close = datos.xs("Close", axis=1, level=1)

        df_largo = df_close.reset_index().melt(
            id_vars="Date", var_name="ticker", value_name="close_ajustado"
        )
        df_largo = df_largo.rename(columns={"Date": "fecha"})
        df_largo = df_largo.dropna(subset=["close_ajustado"])

        dataframes_lotes.append(df_largo)

        if i < len(lotes) - 1:
            time.sleep(PAUSA_ENTRE_LOTES_SEGUNDOS)

    df_final = pd.concat(dataframes_lotes, ignore_index=True)
    df_final = df_final.sort_values(["ticker", "fecha"]).reset_index(drop=True)

    print(f"\nDescarga completada. Shape: {df_final.shape}")
    print(f"Tickers con datos: {df_final['ticker'].nunique()} de {len(tickers)} solicitados")

    return df_final


def guardar_precios_individuales_yahoo() -> None:
    tickers_activos = identificar_tickers_activos()
    df_precios = descargar_precios_individuales_yahoo(tickers_activos)

    ruta_salida = f"{RUTA_DATOS_CRUDOS}/yahoo_precios_individuales.csv"
    df_precios.to_csv(ruta_salida, index=False)
    print(f"Guardado en {ruta_salida}")


if __name__ == "__main__":
    # descargar_todas_series_yahoo()
    guardar_precios_individuales_yahoo()