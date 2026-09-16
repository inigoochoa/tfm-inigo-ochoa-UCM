"""Loader de datos Sharadar: universo S&P500 point-in-time, metadatos de tickers, fundamentales ARQ, y precios de tickers delistados (complemento a Yahoo Finance)."""

import os
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests
from dotenv import load_dotenv

from config.series import (
    URL_BASE_SHARADAR,
    TABLA_SHARADAR_SP500,
    TABLA_SHARADAR_TICKERS,
    TABLA_SHARADAR_FUNDAMENTALS,
    TABLA_SHARADAR_STOCKS,
    RUTA_DATOS_CRUDOS,
)
from data_ingestion.universe import tickers_por_estado, tickers_universo_historico

load_dotenv()

DIMENSION_OBJETIVO = "ARQ"

# Columnas necesarias para el valuation composite (earnings yield +
# cash flow/operating income) y contexto point-in-time. Se filtra
# aquí, no en la API, ya que el bulk trae todo de golpe.
COLUMNAS_NECESARIAS_FUNDAMENTALS = [
    "ticker",
    "dimension",
    "date",           # datekey real de filing -> clave para anti-look-ahead
    "calendardate",
    "reportperiod",
    "lastupdated",
    "netinc",         # net income -> earnings yield
    "netinccmn",       # net income common stock (más preciso para EPS)
    "ebit",
    "ebitda",
    "ncfo",           # net cash flow from operations -> cash flow yield
    "opinc",          # operating income
    "price",          # share price adjusted close, en la fecha del filing
    "marketcap",
    "sharesbas",
    "equity",         # HML (book-to-market) y denominador de RMW
    "intexp",         # RMW: (EBIT - INTEXP) / EQUITY
    "assets",         # CMA (denominador) y ASSETS_(t-1)
    "ppnenet",        # CMA (Δ PP&E neto)
    "depamor",        # CMA (D&A del periodo)
    "inventory",
]

TAMANO_CHUNK_FUNDAMENTALS = 500_000  # filas por chunk al leer el CSV descomprimido

# Precios: solo se descargan para tickers delistados (isdelisted == "Y"
# en sharadar_tickers.csv), que Yahoo Finance no cubre de forma
# fiable. Los tickers activos se descargan vía Yahoo (yahoo_loader.py),
# coherente con el diseño de producción del proyecto.
COLUMNAS_NECESARIAS_STOCKS = ["ticker", "date", "closeadj"]
TAMANO_CHUNK_STOCKS = 1_000_000  # filas por chunk al leer el CSV descomprimido


def _obtener_api_key() -> str:
    api_key = os.getenv("SHARADAR_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SHARADAR_API_KEY no está definida. Copia .env.example a .env y rellénala."
        )
    return api_key


def _descargar_tabla_bulk(nombre_tabla: str) -> pd.DataFrame:
    # Para tablas de tamaño moderado (sp500, tickers); fundamentals y stocks usan
    # _descargar_zip_bulk_* (streaming, dado su tamaño).
    api_key = _obtener_api_key()
    url = f"{URL_BASE_SHARADAR}/{nombre_tabla}"
    params = {"api_key": api_key, "years": "full"}

    print(f"Descargando tabla '{nombre_tabla}' (bulk, histórico completo)...")
    respuesta = requests.get(url, params=params, timeout=60)
    respuesta.raise_for_status()

    with ZipFile(BytesIO(respuesta.content)) as zip_file:
        nombre_csv = zip_file.namelist()[0]
        with zip_file.open(nombre_csv) as csv_file:
            df = pd.read_csv(csv_file)

    print(f"  Descarga completada. Shape: {df.shape}")
    return df


def descargar_sp500() -> pd.DataFrame:
    df = _descargar_tabla_bulk(TABLA_SHARADAR_SP500)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def descargar_tickers() -> pd.DataFrame:
    df = _descargar_tabla_bulk(TABLA_SHARADAR_TICKERS)
    return df


def guardar_universo_sharadar() -> None:
    df_sp500 = descargar_sp500()
    ruta_sp500 = f"{RUTA_DATOS_CRUDOS}/sharadar_sp500.csv"
    df_sp500.to_csv(ruta_sp500, index=False)
    print(f"Guardado en {ruta_sp500}")

    df_tickers = descargar_tickers()
    ruta_tickers = f"{RUTA_DATOS_CRUDOS}/sharadar_tickers.csv"
    df_tickers.to_csv(ruta_tickers, index=False)
    print(f"Guardado en {ruta_tickers}")


def _descargar_zip_bulk_fundamentals() -> BytesIO:
    api_key = _obtener_api_key()
    url = f"{URL_BASE_SHARADAR}/{TABLA_SHARADAR_FUNDAMENTALS}"
    params = {"api_key": api_key, "years": "full"}

    print("Descargando bulk de 'fundamentals' (~620 MB, puede tardar varios minutos)...")
    respuesta = requests.get(url, params=params, timeout=600, stream=True)
    respuesta.raise_for_status()

    buffer = BytesIO()
    descargado = 0
    for chunk in respuesta.iter_content(chunk_size=1024 * 1024):
        buffer.write(chunk)
        descargado += len(chunk)
        print(f"  Descargados {descargado / 1_000_000:.0f} MB...", end="\r")

    print(f"\nDescarga completada: {descargado / 1_000_000:.0f} MB")
    buffer.seek(0)
    return buffer


def descargar_fundamentals_arq() -> pd.DataFrame:
    zip_buffer = _descargar_zip_bulk_fundamentals()

    with ZipFile(zip_buffer) as zip_file:
        nombre_csv = zip_file.namelist()[0]
        print(f"Procesando '{nombre_csv}' por chunks (dimension={DIMENSION_OBJETIVO})...")

        chunks_filtrados = []
        with zip_file.open(nombre_csv) as csv_file:
            lector = pd.read_csv(
                csv_file,
                chunksize=TAMANO_CHUNK_FUNDAMENTALS,
                usecols=lambda col: col in COLUMNAS_NECESARIAS_FUNDAMENTALS,
            )
            for i, chunk in enumerate(lector):
                chunk_arq = chunk[chunk["dimension"] == DIMENSION_OBJETIVO]
                if not chunk_arq.empty:
                    chunks_filtrados.append(chunk_arq)
                print(f"  Chunk {i + 1} procesado ({len(chunk)} filas leídas)...", end="\r")

    df = pd.concat(chunks_filtrados, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    df["reportperiod"] = pd.to_datetime(df["reportperiod"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"\nProcesamiento completado. Shape final (ARQ): {df.shape}")
    return df


def guardar_fundamentals_arq() -> None:
    df = descargar_fundamentals_arq()
    ruta_salida = f"{RUTA_DATOS_CRUDOS}/sharadar_fundamentals_arq.csv"
    df.to_csv(ruta_salida, index=False)
    print(f"Guardado en {ruta_salida}")
    print(f"Rango de fechas (date/filing): {df['date'].min()} a {df['date'].max()}")
    print(f"Tickers únicos: {df['ticker'].nunique()}")


def _descargar_zip_bulk_stocks() -> BytesIO:
    api_key = _obtener_api_key()
    url = f"{URL_BASE_SHARADAR}/{TABLA_SHARADAR_STOCKS}"
    params = {"api_key": api_key, "years": "full"}

    print("Descargando bulk de 'stocks' (puede tardar varios minutos)...")
    respuesta = requests.get(url, params=params, timeout=900, stream=True)
    respuesta.raise_for_status()

    buffer = BytesIO()
    descargado = 0
    for chunk in respuesta.iter_content(chunk_size=1024 * 1024):
        buffer.write(chunk)
        descargado += len(chunk)
        print(f"  Descargados {descargado / 1_000_000:.0f} MB...", end="\r")

    print(f"\nDescarga completada: {descargado / 1_000_000:.0f} MB")
    buffer.seek(0)
    return buffer


def descargar_precios_delistados(tickers_delistados: set[str]) -> pd.DataFrame:
    # closeadj (ajustado por splits/dividendos/spinoffs) es comparable con el
    # "Adj Close" de Yahoo Finance usado para los tickers activos.
    zip_buffer = _descargar_zip_bulk_stocks()

    with ZipFile(zip_buffer) as zip_file:
        nombre_csv = zip_file.namelist()[0]
        print(f"Procesando '{nombre_csv}' por chunks "
              f"(filtrando a {len(tickers_delistados)} tickers delistados)...")

        chunks_filtrados = []
        with zip_file.open(nombre_csv) as csv_file:
            lector = pd.read_csv(
                csv_file,
                chunksize=TAMANO_CHUNK_STOCKS,
                usecols=lambda col: col in COLUMNAS_NECESARIAS_STOCKS,
            )
            for i, chunk in enumerate(lector):
                chunk_filtrado = chunk[chunk["ticker"].isin(tickers_delistados)]
                if not chunk_filtrado.empty:
                    chunks_filtrados.append(chunk_filtrado)
                print(f"  Chunk {i + 1} procesado ({len(chunk)} filas leídas)...", end="\r")

    df = pd.concat(chunks_filtrados, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"\nProcesamiento completado. Shape final: {df.shape}")
    print(f"Tickers encontrados: {df['ticker'].nunique()} de {len(tickers_delistados)} solicitados")

    return df


def identificar_tickers_delistados() -> set[str]:
    tickers_historico = tickers_universo_historico()
    tickers_delistados = tickers_por_estado("Y")

    print(f"Tickers en universo histórico S&P500: {len(tickers_historico)}")
    print(f"De ellos, delistados (requieren Sharadar): {len(tickers_delistados)}")
    print(f"Activos (irán por Yahoo): {len(tickers_historico) - len(tickers_delistados)}")

    return tickers_delistados


def guardar_precios_delistados() -> None:
    tickers_delistados = identificar_tickers_delistados()
    df_precios = descargar_precios_delistados(tickers_delistados)

    ruta_salida = f"{RUTA_DATOS_CRUDOS}/sharadar_precios_delistados.csv"
    df_precios.to_csv(ruta_salida, index=False)
    print(f"Guardado en {ruta_salida}")


if __name__ == "__main__":
    #guardar_universo_sharadar()
    guardar_fundamentals_arq()
    #guardar_precios_delistados()