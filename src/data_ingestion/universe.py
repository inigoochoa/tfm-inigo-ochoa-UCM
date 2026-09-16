"""Construcción del universo de acciones (S&P500 + securities master SF1) a partir de los datos trimestrales de Sharadar. Punto único reutilizado por yahoo_loader.py y sharadar_loader.py."""

from functools import lru_cache

import pandas as pd
from config.series import RUTA_DATOS_CRUDOS

RUTA_SP500 = f"{RUTA_DATOS_CRUDOS}/sharadar_sp500.csv"
RUTA_TICKERS = f"{RUTA_DATOS_CRUDOS}/sharadar_tickers.csv"


TABLA_TICKERS_RELEVANTE = "SF1"


@lru_cache(maxsize=None)
def _cargar_snapshots_historicos(ruta_sp500: str = RUTA_SP500) -> pd.DataFrame:
    df = pd.read_csv(ruta_sp500, parse_dates=["date"])
    df_historico = df[df["action"] == "historical"].copy()
    return df_historico


@lru_cache(maxsize=None)
def _cargar_tickers_sf1(ruta_tickers: str = RUTA_TICKERS) -> pd.DataFrame:
    df_tickers = pd.read_csv(ruta_tickers)
    return df_tickers[df_tickers["table"] == TABLA_TICKERS_RELEVANTE]


def tickers_universo_historico(ruta_sp500: str = RUTA_SP500) -> set[str]:
    # Cualquier action (no solo 'historical'): superset usado para decidir qué
    # tickers necesitan precios individuales. get_universe() sí restringe a
    # 'historical' porque necesita la composición exacta en una fecha concreta.
    df_sp500 = pd.read_csv(ruta_sp500)
    return set(df_sp500["ticker"].unique())


def tickers_por_estado(
    isdelisted: str,
    ruta_sp500: str = RUTA_SP500,
    ruta_tickers: str = RUTA_TICKERS,
) -> set[str]:
    """isdelisted: "Y" para tickers delistados, "N" para activos."""
    tickers_historico = tickers_universo_historico(ruta_sp500)
    df_tickers_sf1 = _cargar_tickers_sf1(ruta_tickers)
    return set(
        df_tickers_sf1[df_tickers_sf1["isdelisted"] == isdelisted]["ticker"]
    ) & tickers_historico


def get_universe(
    fecha: str | pd.Timestamp,
    method: str = "sp500",
    verbose: bool = True,
) -> pd.DataFrame:
    """Universo de acciones point-in-time para una fecha dada. verbose=False para llamadas en bucle (evita imprimir por iteración)."""
    if method != "sp500":
        raise NotImplementedError(
            f"Método de universo '{method}' no implementado. "
            "Solo 'sp500' está disponible actualmente."
        )

    fecha = pd.to_datetime(fecha)
    df_historico = _cargar_snapshots_historicos()

    fechas_snapshot = df_historico["date"].unique()
    fechas_validas = fechas_snapshot[fechas_snapshot <= fecha]

    if len(fechas_validas) == 0:
        raise ValueError(
            f"No hay snapshot disponible para la fecha {fecha.date()}. "
            f"El snapshot más antiguo es {fechas_snapshot.min()}."
        )

    fecha_snapshot = fechas_validas.max()
    tickers_snapshot = df_historico[df_historico["date"] == fecha_snapshot][
        "ticker"
    ].unique()

    df_tickers_sf1 = _cargar_tickers_sf1()

    df_universo = df_tickers_sf1[df_tickers_sf1["ticker"].isin(tickers_snapshot)][
        ["ticker", "name", "sector", "industry"]
    ].drop_duplicates(subset="ticker")

    if verbose:
        print(
            f"Universo para {fecha.date()} (snapshot {fecha_snapshot.date()}): "
            f"{len(df_universo)} tickers"
        )

    return df_universo.reset_index(drop=True)


def construir_calendario_universo(
    fecha_inicio: str | pd.Timestamp,
    fecha_fin: str | pd.Timestamp,
    method: str = "sp500",
) -> pd.DataFrame:
    """Panel largo (ticker, fecha_mes) llamando a get_universe() para cada fin de mes del rango. Meses sin snapshot disponible se omiten con un aviso, en vez de fallar."""
    fines_de_mes = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="ME")

    paneles_mensuales = []
    meses_omitidos = 0
    for fecha_mes in fines_de_mes:
        try:
            df_universo_mes = get_universe(fecha_mes, method=method, verbose=False)
        except ValueError:
            meses_omitidos += 1
            continue
        df_universo_mes = df_universo_mes[["ticker"]].copy()
        df_universo_mes["fecha_mes"] = fecha_mes
        paneles_mensuales.append(df_universo_mes)

    if meses_omitidos > 0:
        print(
            f"Aviso: {meses_omitidos} meses omitidos por no tener snapshot "
            f"de universo disponible (cobertura real de Sharadar sp500 "
            f"empieza más tarde de lo solicitado, ver limitación documentada)."
        )

    df_calendario = pd.concat(paneles_mensuales, ignore_index=True)
    print(
        f"Calendario de universo construido: {len(df_calendario['fecha_mes'].unique())} meses "
        f"(de {len(fines_de_mes)} solicitados), "
        f"{df_calendario['ticker'].nunique()} tickers únicos, "
        f"{len(df_calendario)} filas ticker-mes"
    )
    return df_calendario.sort_values(["ticker", "fecha_mes"]).reset_index(drop=True)

if __name__ == "__main__":
    universo_ejemplo = get_universe("2015-06-30")
    print(universo_ejemplo.head(10))
    print(f"Total tickers: {len(universo_ejemplo)}")