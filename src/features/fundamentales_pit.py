"""Merge point-in-time entre el universo mensual y los fundamentales trimestrales de Sharadar, sin look-ahead: usa el último filing publicado hasta cada fecha."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from data_ingestion.universe import construir_calendario_universo

DIMENSION_OBJETIVO = "ARQ"
TOLERANCIA_DIAS_FILING = 189  # ~2 trimestres
NOMBRE_FICHERO_PIT = "fundamentales_pit.parquet"


def _preparar_fundamentales(
    fundamentales: pd.DataFrame,
    dimension: str = DIMENSION_OBJETIVO,
) -> pd.DataFrame:
    df = fundamentales.loc[fundamentales["dimension"] == dimension].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["reportperiod"] = pd.to_datetime(df["reportperiod"])

    if "lastupdated" in df.columns:
        df["lastupdated"] = pd.to_datetime(df["lastupdated"])
        df = df.sort_values(["ticker", "date", "lastupdated"])
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def fundamentales_point_in_time(
    fundamentales: pd.DataFrame,
    fecha_inicio: str | pd.Timestamp,
    fecha_fin: str | pd.Timestamp,
    dimension: str = DIMENSION_OBJETIVO,
    tolerancia_dias: int = TOLERANCIA_DIAS_FILING,
) -> pd.DataFrame:
    # merge_asof backward + tolerancia: si no hay filing dentro de la ventana,
    # la empresa queda excluida ese mes (NaN) en vez de rellenarse indefinidamente.
    fund_prep = _preparar_fundamentales(fundamentales, dimension=dimension)
    calendario = construir_calendario_universo(fecha_inicio, fecha_fin)

    # merge_asof con 'by' exige que 'on' esté ordenada globalmente, no por ticker+fecha.
    fund_prep_merge = fund_prep.sort_values("date").reset_index(drop=True)
    calendario_merge = calendario.sort_values("fecha_mes").reset_index(drop=True)

    resultado = pd.merge_asof(
        calendario_merge,
        fund_prep_merge,
        left_on="fecha_mes",
        right_on="date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerancia_dias),
    )

    resultado = resultado.rename(columns={"date": "fecha_filing"})

    n_excluidas = resultado["fecha_filing"].isna().sum()
    n_total = len(resultado)
    print(
        f"Merge point-in-time completado: {n_total} filas ticker-mes, "
        f"{n_excluidas} excluidas por filing ausente o >{tolerancia_dias} días "
        f"({n_excluidas / n_total:.1%})"
    )

    return resultado


def guardar_fundamentales_pit(
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
    ruta_fundamentales = "data/raw/sharadar_fundamentals_arq.csv"
    df_fundamentales = pd.read_csv(ruta_fundamentales)

    resultado = fundamentales_point_in_time(
        fundamentales=df_fundamentales,
        fecha_inicio="1990-01-01",
        fecha_fin="2026-06-30",
    )

    print(resultado.head(10))

    ejemplo = resultado[resultado["ticker"] == "AAPL"].tail(10)
    print("\nEjemplo AAPL (últimos 10 meses):")
    print(ejemplo[["ticker", "fecha_mes", "fecha_filing", "netinc", "marketcap"]])

    guardar_fundamentales_pit(resultado)