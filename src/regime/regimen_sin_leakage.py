"""Corrige el look-ahead de kmeans_regimen.py (centroides ajustados sobre todo el panel) prediciendo 'regimen_predicho' con un RF entrenado solo hasta FECHA_CORTE_TRAIN. Capa 2 debe usar 'regimen_predicho', no 'regimen'."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from regime.rf_clasificador_regimen import (
    FECHA_CORTE_TRAIN,
    cargar_panel_con_regimen,
    dividir_train_test,
    entrenar_rf_regimen,
    preparar_datos_rf,
)

NOMBRE_FICHERO_PANEL_CON_REGIMEN = "panel_mensual_con_regimen.csv"


def generar_regimen_sin_leakage() -> pd.DataFrame:
    panel = cargar_panel_con_regimen()
    df_modelo = preparar_datos_rf(panel)

    X_train, X_test, y_train, y_test = dividir_train_test(df_modelo, FECHA_CORTE_TRAIN)
    modelo = entrenar_rf_regimen(X_train, y_train)

    X_completo = df_modelo[X_train.columns]
    df_modelo["regimen_predicho"] = modelo.predict(X_completo)

    n_coincide = (df_modelo["regimen_predicho"] == df_modelo["regimen"]).mean()
    print(f"Coincidencia regimen_predicho vs regimen (k-means): {n_coincide:.1%} "
          f"(el desacuerdo es esperado, sobre todo en test -- el RF nunca alcanza "
          f"F1 perfecto, ver validación de Capa 1: F1 macro 0.4728)")

    panel_actualizado = panel.join(df_modelo[["regimen_predicho"]], how="left")

    n_sin_prediccion = panel_actualizado["regimen_predicho"].isna().sum()
    if n_sin_prediccion > 0:
        print(f"Aviso: {n_sin_prediccion} filas sin regimen_predicho "
              f"(NaN en alguna FEATURES_REGIMEN, excluidas también de 'regimen' original).")

    return panel_actualizado


def guardar_panel_con_regimen_predicho(panel: pd.DataFrame) -> None:
    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PANEL_CON_REGIMEN}"
    panel.to_csv(ruta_salida)
    print(f"Guardado en {ruta_salida} (columna 'regimen_predicho' añadida)")


if __name__ == "__main__":
    panel = generar_regimen_sin_leakage()
    print(panel[["regimen", "regimen_predicho"]].tail(10))
    guardar_panel_con_regimen_predicho(panel)