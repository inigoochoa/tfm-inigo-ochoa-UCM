"""
Consolida las 5 fuentes de series temporales (FRED, Yahoo Finance,
Kenneth French, AQR BAB, AAII) en un único panel mensual maestro.
"""

import pandas as pd
from config.series import FECHA_INICIO, RUTA_DATOS_CRUDOS, RUTA_DATOS_PROCESADOS


COLUMNAS_MEDIA_MENSUAL = {
    "vix",
    "peticiones_paro",
    "bullish",
    "neutral",
    "bearish",
}


def _cargar_csv_fecha(nombre_fichero: str) -> pd.DataFrame:
    """Carga un CSV de data/raw con índice 'fecha' como datetime."""
    ruta = f"{RUTA_DATOS_CRUDOS}/{nombre_fichero}"
    df = pd.read_csv(ruta, index_col="fecha", parse_dates=True)
    return df


def cargar_macro_fred() -> pd.DataFrame:
    return _cargar_csv_fecha("macro_fred.csv")


def cargar_mercado_yahoo() -> pd.DataFrame:
    return _cargar_csv_fecha("mercado_yahoo.csv")


def cargar_factores_kenneth_french() -> pd.DataFrame:
    return _cargar_csv_fecha("factores_kenneth_french.csv")


def cargar_bab_aqr() -> pd.DataFrame:
    return _cargar_csv_fecha("bab_aqr.csv")


def cargar_aaii() -> pd.DataFrame:
    return _cargar_csv_fecha("aaii_sentiment.csv")


def resamplear_a_mensual(df: pd.DataFrame) -> pd.DataFrame:
    columnas_media_presentes = [c for c in df.columns if c in COLUMNAS_MEDIA_MENSUAL]
    columnas_last = [c for c in df.columns if c not in COLUMNAS_MEDIA_MENSUAL]

    partes = []
    if columnas_last:
        partes.append(df[columnas_last].resample("ME").last())
    if columnas_media_presentes:
        partes.append(df[columnas_media_presentes].resample("ME").mean())

    df_mensual = pd.concat(partes, axis=1)
    df_mensual = df_mensual[df.columns]
    return df_mensual


def construir_panel_mensual() -> pd.DataFrame:
    fuentes = {
        "FRED": cargar_macro_fred(),
        "Yahoo Finance": cargar_mercado_yahoo(),
        "Kenneth French": cargar_factores_kenneth_french(),
        "AQR BAB": cargar_bab_aqr(),
        "AAII": cargar_aaii(),
    }

    dataframes_mensuales = []
    for nombre_fuente, df in fuentes.items():
        df_mensual = resamplear_a_mensual(df)
        print(f"{nombre_fuente}: {df.shape} -> {df_mensual.shape} tras resample mensual")
        dataframes_mensuales.append(df_mensual)

    panel = pd.concat(dataframes_mensuales, axis=1, join="outer")
    panel = panel.sort_index()
    panel = panel.loc[FECHA_INICIO:]

    print(f"\nPanel mensual combinado: {panel.shape}")
    print(f"Rango de fechas: {panel.index.min()} a {panel.index.max()}")
    print(f"\nNulos por columna:\n{panel.isna().sum()}")

    return panel


def guardar_panel_mensual() -> None:
    panel = construir_panel_mensual()
    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/panel_mensual.csv"
    panel.to_csv(ruta_salida)
    print(f"\nGuardado en {ruta_salida}")


if __name__ == "__main__":
    guardar_panel_mensual()