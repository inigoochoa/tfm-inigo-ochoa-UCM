"""Merge point-in-time entre el universo mensual y los precios por ticker, combinando delistados (Sharadar) y activos (Yahoo) para evitar survivorship bias."""

import pandas as pd

from config.series import RUTA_DATOS_CRUDOS, RUTA_DATOS_PROCESADOS
from data_ingestion.universe import construir_calendario_universo
from data_ingestion.yahoo_loader import convertir_ticker_yahoo_a_sharadar

NOMBRE_FICHERO_PIT = "precios_pit.parquet"
NOMBRE_COLUMNA_PRECIO = "precio_ajustado"


def _cargar_precios_sharadar(
    ruta_directorio: str = RUTA_DATOS_CRUDOS,
) -> pd.DataFrame:
    ruta = f"{ruta_directorio}/sharadar_precios_delistados.csv"
    df = pd.read_csv(ruta, parse_dates=["date"])
    df = df.rename(columns={"closeadj": NOMBRE_COLUMNA_PRECIO})
    return df[["ticker", "date", NOMBRE_COLUMNA_PRECIO]]


def _cargar_precios_yahoo(
    ruta_directorio: str = RUTA_DATOS_CRUDOS,
) -> pd.DataFrame:
    ruta = f"{ruta_directorio}/yahoo_precios_individuales.csv"
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df["ticker"] = df["ticker"].apply(convertir_ticker_yahoo_a_sharadar)
    df = df.rename(columns={"fecha": "date", "close_ajustado": NOMBRE_COLUMNA_PRECIO})
    return df[["ticker", "date", NOMBRE_COLUMNA_PRECIO]]


def combinar_fuentes_precios(
    ruta_directorio: str = RUTA_DATOS_CRUDOS,
) -> pd.DataFrame:
    precios_sharadar = _cargar_precios_sharadar(ruta_directorio)
    precios_yahoo = _cargar_precios_yahoo(ruta_directorio)

    tickers_solapados = set(precios_sharadar["ticker"]) & set(precios_yahoo["ticker"])
    if tickers_solapados:
        print(
            f"AVISO: {len(tickers_solapados)} tickers aparecen en ambas fuentes "
            f"(Sharadar y Yahoo) -- probablemente cambiaron de estado (delistado/activo) "
            f"entre descargas. Se conservan ambos registros, sin deduplicar. "
            f"Tickers: {sorted(tickers_solapados)[:10]}{'...' if len(tickers_solapados) > 10 else ''}"
        )

    combinado = pd.concat([precios_sharadar, precios_yahoo], ignore_index=True)
    combinado = combinado.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(
        f"Fuentes combinadas: {len(precios_sharadar)} filas Sharadar (delistados) + "
        f"{len(precios_yahoo)} filas Yahoo (activos) = {len(combinado)} filas totales, "
        f"{combinado['ticker'].nunique()} tickers únicos."
    )

    return combinado


def agregar_precios_mensual(precios_diarios: pd.DataFrame) -> pd.DataFrame:
    # Último precio de cotización disponible en cada mes (no el de fin de mes
    # calendario exacto, que puede no ser día de cotización).
    df = precios_diarios.set_index("date")

    mensual = (
        df.groupby("ticker")[NOMBRE_COLUMNA_PRECIO]
        .resample("ME")
        .last()
        .reset_index()
    )
    mensual = mensual.rename(columns={"date": "fecha_mes"})
    mensual = mensual.dropna(subset=[NOMBRE_COLUMNA_PRECIO])

    return mensual


def precios_point_in_time(
    precios_diarios: pd.DataFrame,
    fecha_inicio: str | pd.Timestamp,
    fecha_fin: str | pd.Timestamp,
) -> pd.DataFrame:
    # Merge exacto por (ticker, fecha_mes): a diferencia de fundamentales_pit.py, los
    # precios no tienen desfase de publicación, así que no hace falta merge_asof.
    precios_mensuales = agregar_precios_mensual(precios_diarios)
    calendario = construir_calendario_universo(fecha_inicio, fecha_fin)

    resultado = calendario.merge(
        precios_mensuales,
        on=["ticker", "fecha_mes"],
        how="left",
    )

    n_sin_precio = resultado[NOMBRE_COLUMNA_PRECIO].isna().sum()
    n_total = len(resultado)
    print(
        f"Merge point-in-time completado: {n_total} filas ticker-mes, "
        f"{n_sin_precio} sin precio disponible ese mes "
        f"({n_sin_precio / n_total:.1%})"
    )

    return resultado


def guardar_precios_pit(
    resultado: pd.DataFrame,
    ruta_directorio: str = RUTA_DATOS_PROCESADOS,
    nombre_fichero: str = NOMBRE_FICHERO_PIT,
) -> None:
    ruta_salida = f"{ruta_directorio}/{nombre_fichero}"
    resultado.to_parquet(ruta_salida, index=False)
    print(f"Guardado en {ruta_salida}")
    print(f"Shape: {resultado.shape}")
    print(
        f"Rango de fechas (fecha_mes): {resultado['fecha_mes'].min().date()} "
        f"a {resultado['fecha_mes'].max().date()}"
    )


if __name__ == "__main__":
    precios_diarios = combinar_fuentes_precios()

    resultado = precios_point_in_time(
        precios_diarios=precios_diarios,
        fecha_inicio="1990-01-01",
        fecha_fin="2026-06-30",  # mismo límite que fundamentales_pit.py y panel_maestro_capa2
    )

    print(resultado.head(10))

    ejemplo = resultado[resultado["ticker"] == "AAPL"].tail(10)
    print("\nEjemplo AAPL (últimos 10 meses):")
    print(ejemplo[["ticker", "fecha_mes", NOMBRE_COLUMNA_PRECIO]])

    guardar_precios_pit(resultado)