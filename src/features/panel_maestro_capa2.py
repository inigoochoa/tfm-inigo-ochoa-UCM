"""Une el panel de régimen/macro (Capa 1) con las señales cross-sectional de mercado (Capa 2) en el panel maestro mensual de entrada a los Random Forest."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS

FECHA_INICIO_CAPA2 = "1998-03-31"
FECHA_FIN_CAPA2 = "2026-06-30"


def cargar_panel_regimen() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_con_regimen.csv"
    return pd.read_csv(ruta, parse_dates=["fecha"]).rename(columns={"fecha": "fecha_mes"})


def cargar_panel_senales_mercado() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_senales_mercado_capa2.csv"
    return pd.read_csv(ruta, parse_dates=["fecha_mes"])


def construir_panel_maestro_capa2(
    fecha_inicio: str = FECHA_INICIO_CAPA2,
    fecha_fin: str = FECHA_FIN_CAPA2,
) -> pd.DataFrame:
    panel_regimen = cargar_panel_regimen()
    panel_mercado = cargar_panel_senales_mercado()

    panel = panel_regimen.merge(panel_mercado, on="fecha_mes", how="inner")
    panel = panel[
        (panel["fecha_mes"] >= fecha_inicio) & (panel["fecha_mes"] <= fecha_fin)
    ].reset_index(drop=True)

    print(f"Panel maestro Capa 2: {panel.shape}, "
          f"{panel['fecha_mes'].min().date()} a {panel['fecha_mes'].max().date()}")

    return panel


def guardar_panel_maestro_capa2(panel: pd.DataFrame) -> None:
    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/panel_maestro_capa2.csv"
    panel.to_csv(ruta_salida, index=False)
    print(f"Guardado en {ruta_salida}")


if __name__ == "__main__":
    panel = construir_panel_maestro_capa2()
    print(panel.isna().sum().sort_values(ascending=False).head(15))
    guardar_panel_maestro_capa2(panel)