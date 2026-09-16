"""Loaders de fuentes secundarias (descargar -> limpiar -> guardar): AQR (BAB), Kenneth French (Fama-French) y AAII (sentimiento)."""

import io
import zipfile
from io import BytesIO

import pandas as pd
import requests

from config.series import (
    DATASETS_KENNETH_FRENCH,
    RUTA_DATOS_CRUDOS,
    URL_AAII_SENTIMENT,
    URL_AQR_BAB,
)

# --- AQR: factor Betting-Against-Beta ---

FILA_HEADER_AQR_BAB = 18


def descargar_bab(
    regiones: list[str] = ["USA"],
    ruta_salida: str = f"{RUTA_DATOS_CRUDOS}/bab_aqr.csv",
) -> pd.DataFrame:
    respuesta = requests.get(URL_AQR_BAB)
    respuesta.raise_for_status()
    df = pd.read_excel(
        BytesIO(respuesta.content),
        sheet_name="BAB Factors",
        header=FILA_HEADER_AQR_BAB,
    )

    columnas = ["DATE"] + regiones
    df = df[columnas].copy()
    df["DATE"] = pd.to_datetime(df["DATE"]) + pd.offsets.MonthEnd(0)
    df = df.set_index("DATE")
    df.index.name = "fecha"
    df = df.rename(columns={r: f"bab_{r.lower()}" for r in regiones})

    df.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}. Shape final: {df.shape}")
    return df


# --- Kenneth French: factores Fama-French ---

URL_BASE_KENNETH_FRENCH = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
)


def _descargar_dataset_kenneth_french(dataset: str) -> pd.DataFrame:
    url = f"{URL_BASE_KENNETH_FRENCH}/{dataset}_CSV.zip"
    respuesta = requests.get(url)
    respuesta.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(respuesta.content)) as zf:
        nombre_csv = zf.namelist()[0]
        with zf.open(nombre_csv) as f:
            contenido = f.read().decode("latin-1")

    fila_header = _encontrar_inicio_datos(contenido)
    df = pd.read_csv(io.StringIO(contenido), skiprows=fila_header, index_col=0)

    return df


def _limpiar_datos_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    es_fila_mensual = df.index.str.match(r"^\d{6}$")
    df = df[es_fila_mensual]

    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "fecha"

    df.columns = df.columns.str.strip()
    df = df.astype(float) / 100

    return df


def _encontrar_inicio_datos(contenido: str) -> int:
    lineas = contenido.splitlines()
    for i, linea in enumerate(lineas):
        primer_campo = linea.split(",")[0].strip()
        if pd.Series([primer_campo]).str.match(r"^\d{6}$").iloc[0]:
            return i - 1
    raise ValueError("No se encontró el inicio de los datos mensuales...")


def descargar_todas_series_kenneth_french(
    ruta_salida: str = f"{RUTA_DATOS_CRUDOS}/factores_kenneth_french.csv",
) -> pd.DataFrame:
    dataframes = []
    for dataset, nombre in DATASETS_KENNETH_FRENCH.items():
        print(f"Descargando {nombre}: ({dataset})...")
        df_crudo = _descargar_dataset_kenneth_french(dataset)
        df_limpio = _limpiar_datos_mensuales(df_crudo)
        dataframes.append(df_limpio)

    df_combinado = pd.concat(dataframes, axis=1)
    df_combinado.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}. Shape final: {df_combinado.shape }")

    return df_combinado


# --- AAII: encuesta de sentimiento ---

URL_AAII_PAGINA_PRINCIPAL = "https://www.aaii.com/sentimentsurvey"
FILA_HEADER_AAII = 3
COLUMNAS_SENTIMIENTO = ["Bullish", "Neutral", "Bearish"]

HEADERS_NAVEGADOR = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

HEADERS_DESCARGA_ARCHIVO = {
    **HEADERS_NAVEGADOR,
    "Referer": URL_AAII_PAGINA_PRINCIPAL,
    "Accept": (
        "application/vnd.ms-excel,application/octet-stream,*/*;q=0.8"
    ),
}


def descargar_aaii(
    ruta_salida: str = f"{RUTA_DATOS_CRUDOS}/aaii_sentiment.csv",
) -> pd.DataFrame:
    with requests.Session() as sesion:
        sesion.headers.update(HEADERS_NAVEGADOR)

        # Se visita la página principal primero para obtener cookies de sesión
        # válidas, simulando navegación real antes de pedir el fichero directamente.
        print(f"Visitando página principal: {URL_AAII_PAGINA_PRINCIPAL}")
        resp_pagina = sesion.get(URL_AAII_PAGINA_PRINCIPAL, timeout=30)
        print(f"  Status: {resp_pagina.status_code}")

        print(f"Descargando encuesta AAII desde: {URL_AAII_SENTIMENT}")
        respuesta = sesion.get(
            URL_AAII_SENTIMENT, headers=HEADERS_DESCARGA_ARCHIVO, timeout=30
        )
        respuesta.raise_for_status()

        content_type = respuesta.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise RuntimeError(
                "AAII devolvió HTML en vez del .xls (bloqueo anti-bot "
                "persistente). Primeros 200 bytes: "
                f"{respuesta.content[:200]}"
            )

    df = pd.read_excel(
        BytesIO(respuesta.content),
        sheet_name="SENTIMENT",
        header=FILA_HEADER_AAII,
        na_values=["#N/A"],
    )

    columnas = ["Date"] + COLUMNAS_SENTIMIENTO
    df = df[columnas].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.dropna(subset=COLUMNAS_SENTIMIENTO)

    df = df.set_index("Date").sort_index()
    df.index.name = "fecha"
    df = df.rename(columns={c: c.lower() for c in COLUMNAS_SENTIMIENTO})

    df.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}. Shape final: {df.shape}")
    print(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
    print(f"Valores nulos por columna:\n{df.isna().sum()}")

    return df


if __name__ == "__main__":
    descargar_bab()
    descargar_todas_series_kenneth_french()
    descargar_aaii()
