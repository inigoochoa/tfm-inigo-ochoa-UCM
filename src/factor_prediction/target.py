"""Target de Capa 2: retorno futuro acumulado de los 7 factores objetivo, a horizontes corto=1m/medio=3m/largo=12m. El look-ahead aquí es intencional (es la definición del target)."""

import numpy as np
import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS

FACTORES_OBJETIVO = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "bab_usa"]

HORIZONTES_MESES = {
    "corto": 1,
    "medio": 3,
    "largo": 12,
}

NOMBRE_FICHERO_PANEL_MAESTRO = "panel_maestro_capa2.csv"


def cargar_factores_desde_panel_maestro(
    ruta_directorio: str = RUTA_DATOS_PROCESADOS,
) -> pd.DataFrame:
    # Se lee del panel maestro (ya alineado, ver alineacion.py) en vez de releer
    # factores_kenneth_french.csv + bab_usa por separado con un resampleo distinto.
    ruta = f"{ruta_directorio}/{NOMBRE_FICHERO_PANEL_MAESTRO}"
    df = pd.read_csv(ruta, parse_dates=["fecha_mes"])
    columnas = ["fecha_mes"] + FACTORES_OBJETIVO
    faltantes = set(columnas) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de factores en el panel maestro: {faltantes}")
    return df[columnas].sort_values("fecha_mes").reset_index(drop=True)


def _retorno_acumulado_futuro(serie: pd.Series, horizonte: int) -> pd.Series:
    """Retorno acumulado compuesto de los `horizonte` meses siguientes a cada fecha t (t+1..t+h). Los últimos `horizonte` valores son NaN por construcción."""
    log_retorno = np.log1p(serie)
    suma_log_futura = log_retorno.rolling(window=horizonte).sum().shift(-horizonte)
    return np.expm1(suma_log_futura)


def construir_targets_por_horizonte(
    df_factores: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    resultado = {}

    for nombre_horizonte, meses in HORIZONTES_MESES.items():
        df_target = pd.DataFrame({"fecha_mes": df_factores["fecha_mes"]})

        for factor in FACTORES_OBJETIVO:
            df_target[f"target_{factor}"] = _retorno_acumulado_futuro(
                df_factores[factor], meses
            )

        n_validos = df_target.drop(columns="fecha_mes").notna().all(axis=1).sum()
        print(
            f"Horizonte '{nombre_horizonte}' ({meses}m): {len(df_target)} filas totales, "
            f"{n_validos} con target completo en los 7 factores "
            f"({len(df_target) - n_validos} sin target por cola de horizonte o NaN de origen)."
        )

        resultado[nombre_horizonte] = df_target

    return resultado


def guardar_targets_por_horizonte(
    targets: dict[str, pd.DataFrame],
    ruta_directorio: str = RUTA_DATOS_PROCESADOS,
) -> None:
    # Un CSV por horizonte (cada uno alimenta un Random Forest independiente),
    # en vez de uno solo con prefijos de columna.
    for nombre_horizonte, df_target in targets.items():
        ruta_salida = f"{ruta_directorio}/target_{nombre_horizonte}.csv"
        df_target.to_csv(ruta_salida, index=False)
        print(f"Guardado {ruta_salida} — shape {df_target.shape}")


if __name__ == "__main__":
    df_factores = cargar_factores_desde_panel_maestro()
    targets = construir_targets_por_horizonte(df_factores)

    print("\nEjemplo target 'medio' (3 meses), primeras y últimas filas:")
    print(targets["medio"].head(5).to_string(index=False))
    print("...")
    print(targets["medio"].tail(5).to_string(index=False))

    guardar_targets_por_horizonte(targets)