"""Descarga series macro de FRED evitando look-ahead: series estándar usan el vintage actual, series revisadas (SERIES_PIT_NIVEL/VAR_PCT) usan el valor point-in-time de ALFRED, con el vintage estándar como fallback antes del inicio de cobertura de ALFRED."""

import os
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from config.series import FECHA_INICIO, FECHA_FIN, SERIES_FRED, RUTA_DATOS_CRUDOS
load_dotenv()

SERIES_ESTANDAR = {"DGS10", "DGS2", "FEDFUNDS", "BAA10Y", "ICSA"}
SERIES_PIT_NIVEL = {"UNRATE", "PSAVERT", "CFNAI"}
SERIES_PIT_VAR_PCT = {"CPIAUCSL", "INDPRO", "PPIACO"}


def _obtener_cliente_fred() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY no está definida. Copia .env.example a .env y rellénala."
        )
    return Fred(api_key=api_key)


def descargar_serie_fred_estandar(codigo_fred: str, nombre_columna: str) -> pd.DataFrame:
    fred = _obtener_cliente_fred()
    serie = fred.get_series(
        codigo_fred,
        observation_start=FECHA_INICIO,
        observation_end=FECHA_FIN,
    )
    df = serie.to_frame(name=nombre_columna)
    df.index.name = "fecha"
    return df


def _obtener_revisiones_fred(codigo_fred: str) -> pd.DataFrame:
    fred = _obtener_cliente_fred()
    df = fred.get_series_all_releases(codigo_fred)
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df


def _construir_pit_nivel_con_fecha(codigo_fred: str) -> pd.DataFrame:
    # Para cada periodo económico (date), la primera publicación conocida
    # (realtime_start mínimo).
    df_revisiones = _obtener_revisiones_fred(codigo_fred)
    idx_primera_publicacion = df_revisiones.groupby("date")["realtime_start"].idxmin()
    df_pit = df_revisiones.loc[idx_primera_publicacion, ["realtime_start", "date", "value"]]
    return df_pit.reset_index(drop=True)


def calcular_variacion_pct_vintage_consistente(df_revisiones: pd.DataFrame) -> pd.DataFrame:
    # Para cada fecha t, usa el valor de t y t-1 tal como estaban publicados en la
    # primera publicación de t (mismo vintage), evitando saltos por rebasing de índice.
    df = df_revisiones.sort_values("realtime_start")

    primera_pub = df.loc[df.groupby("date")["realtime_start"].idxmin()]
    primera_pub = primera_pub.sort_values("date").reset_index(drop=True)

    resultados = []
    for _, fila in primera_pub.iterrows():
        fecha_t = fila["date"]
        v_t = fila["realtime_start"]
        valor_t = fila["value"]

        fecha_t_menos_1 = fecha_t - pd.DateOffset(months=1)

        candidatos = df[
            (df["date"] == fecha_t_menos_1) & (df["realtime_start"] <= v_t)
        ]
        if candidatos.empty:
            continue
        valor_t_menos_1 = candidatos.loc[candidatos["realtime_start"].idxmax(), "value"]

        if pd.notna(valor_t) and pd.notna(valor_t_menos_1) and valor_t_menos_1 != 0:
            resultados.append({"date": fecha_t, "realtime_start": v_t, "value": valor_t / valor_t_menos_1 - 1})

    return pd.DataFrame(resultados)


def _resamplear_pit_a_mensual(
    df_pit_con_fecha: pd.DataFrame, nombre_columna: str
) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = df_pit_con_fecha.sort_values("realtime_start").copy()
    # fecha_corte = primer realtime_start real (día en que ALFRED empezó a trackear
    # la serie), NUNCA el "date" (periodo económico) de esa fila: ALFRED suele
    # backfillear periodos históricos muy anteriores en su primer día de tracking.
    fecha_corte = df["realtime_start"].min()

    df_indexado = df.set_index("realtime_start")[["value"]]
    df_indexado.index.name = "fecha"
    df_mensual = df_indexado.resample("ME").last()
    df_mensual[f"{nombre_columna}_es_pit_real"] = df_mensual["value"].notna()
    df_mensual[nombre_columna] = df_mensual["value"].astype(float).ffill()
    df_mensual = df_mensual.drop(columns="value")

    return df_mensual, fecha_corte


def _construir_serie_hibrida(
    codigo_fred: str,
    nombre_columna: str,
    df_pit_con_fecha: pd.DataFrame,
) -> pd.DataFrame:
    df_pit_mensual, fecha_corte = _resamplear_pit_a_mensual(df_pit_con_fecha, nombre_columna)

    df_estandar = descargar_serie_fred_estandar(codigo_fred, nombre_columna)
    df_estandar_temprano = df_estandar.loc[df_estandar.index < fecha_corte].copy()
    df_estandar_temprano[f"{nombre_columna}_es_pit_real"] = False

    df_hibrido = pd.concat([df_estandar_temprano, df_pit_mensual])
    df_hibrido = df_hibrido.sort_index()
    df_hibrido.index.name = "fecha"
    df_hibrido = df_hibrido.loc[FECHA_INICIO:FECHA_FIN]

    if len(df_estandar_temprano) > 0:
        print(
            f"  Aviso: {codigo_fred} usa vintage estándar (fallback) "
            f"antes de {fecha_corte.date()} ({len(df_estandar_temprano)} "
            f"observaciones), por falta de cobertura ALFRED genuina."
        )

    return df_hibrido


def descargar_serie_fred_pit_nivel(codigo_fred: str, nombre_columna: str) -> pd.DataFrame:
    df_pit_con_fecha = _construir_pit_nivel_con_fecha(codigo_fred)
    return _construir_serie_hibrida(codigo_fred, nombre_columna, df_pit_con_fecha)


def descargar_serie_fred_pit_variacion(codigo_fred: str, nombre_columna: str) -> pd.DataFrame:
    nombre_columna_var = f"{nombre_columna}_var_pct"

    df_revisiones = _obtener_revisiones_fred(codigo_fred)
    df_var_con_fecha = calcular_variacion_pct_vintage_consistente(df_revisiones)

    df_pit_mensual, fecha_corte = _resamplear_pit_a_mensual(df_var_con_fecha, nombre_columna_var)

    df_estandar_nivel = descargar_serie_fred_estandar(codigo_fred, nombre_columna)
    df_estandar_var = df_estandar_nivel.pct_change(fill_method=None).rename(
        columns={nombre_columna: nombre_columna_var}
    )
    df_estandar_temprano = df_estandar_var.loc[df_estandar_var.index < fecha_corte].copy()
    df_estandar_temprano[f"{nombre_columna_var}_es_pit_real"] = False

    df_hibrido = pd.concat([df_estandar_temprano, df_pit_mensual])
    df_hibrido = df_hibrido.sort_index()
    df_hibrido.index.name = "fecha"
    df_hibrido = df_hibrido.loc[FECHA_INICIO:FECHA_FIN]

    if len(df_estandar_temprano) > 0:
        print(
            f"  Aviso: {codigo_fred} usa vintage estándar (fallback) "
            f"antes de {fecha_corte.date()} ({len(df_estandar_temprano)} "
            f"observaciones), por falta de cobertura ALFRED genuina."
        )

    return df_hibrido


def descargar_todas_series_fred() -> pd.DataFrame:
    dataframes = []
    for codigo_fred, nombre_columna in SERIES_FRED.items():
        if codigo_fred in SERIES_PIT_VAR_PCT:
            print(f"Descargando {codigo_fred} -> {nombre_columna}_var_pct (PIT híbrida, variación)...")
            df_serie = descargar_serie_fred_pit_variacion(codigo_fred, nombre_columna)
        elif codigo_fred in SERIES_PIT_NIVEL:
            print(f"Descargando {codigo_fred} -> {nombre_columna} (PIT híbrida, nivel)...")
            df_serie = descargar_serie_fred_pit_nivel(codigo_fred, nombre_columna)
        else:
            print(f"Descargando {codigo_fred} -> {nombre_columna} (estándar)...")
            df_serie = descargar_serie_fred_estandar(codigo_fred, nombre_columna)
        dataframes.append(df_serie)

    df_combinado = pd.concat(dataframes, axis=1)
    return df_combinado


if __name__ == "__main__":
    df_macro = descargar_todas_series_fred()
    print(df_macro.head())
    print(df_macro.tail())
    print(f"Filas: {len(df_macro)}, columnas: {list(df_macro.columns)}")

    ruta_salida = f"{RUTA_DATOS_CRUDOS}/macro_fred.csv"
    df_macro.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida}")