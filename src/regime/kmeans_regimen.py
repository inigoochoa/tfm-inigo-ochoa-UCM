"""Clustering k-means (k=4) de régimen de mercado sobre variables macro/de mercado, excluyendo los factores objetivo y sus señales cross-sectional (pertenecen a Capa 2)."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config.series import RUTA_DATOS_PROCESADOS

# Excluye los 7 factores objetivo y sus señales cross-sectional (rs_*), que
# pertenecen a Capa 2. Los niveles brutos de tipos/tasas usan su z-score móvil de
# 60m en vez del nivel bruto, para no confundir deriva secular con ciclo económico.
FEATURES_REGIMEN = [
    "tipo_10y_zscore_60m",
    "tipo_2y_zscore_60m",
    "fed_funds_zscore_60m",
    "spread_credito",
    "cpi_var_pct",
    "produccion_industrial_var_pct",
    "tasa_paro_zscore_60m",
    "ppi_var_pct",
    "tasa_ahorro_zscore_60m",
    "peticiones_paro",
    "indice_ciclo",
    "retorno_sp500",
    "vix",
    "spread_10y_2y",
    "drawdown_sp500_12m",
    "vol_realizada_sp500",
    "bullish",
    "neutral",
    "bearish",
]

# log1p antes de escalar: el pico de covid (~4.66M) es un outlier aislado que
# infla la std usada por StandardScaler y "silenciaría" la variable sin transformar.
FEATURES_LOG1P = {"peticiones_paro"}

K_REGIMENES = 4
SEMILLA_ALEATORIA = 42


def cargar_panel_features_macro() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_features_completo.csv"
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)


def preparar_datos_clustering(panel: pd.DataFrame) -> pd.DataFrame:
    # Se recorta a la primera fecha sin NaN en vez de imputar: los NaN aquí
    # afectarían a la propia definición del régimen, no a un feature de predicción.
    df_features = panel[FEATURES_REGIMEN].copy()

    for columna in FEATURES_LOG1P:
        df_features[columna] = np.log1p(df_features[columna])

    primera_fecha_completa = df_features.dropna().index.min()

    print(f"Primera fecha con las {len(FEATURES_REGIMEN)} features completas: "
          f"{primera_fecha_completa.date()}")

    df_recortado = df_features.loc[primera_fecha_completa:]

    nulos_restantes = df_recortado.isna().sum().sum()
    if nulos_restantes > 0:
        print(f"Aviso: quedan {nulos_restantes} NaN dispersos tras el recorte "
              f"inicial (huecos intermedios) -- se eliminan las filas afectadas.")
        df_recortado = df_recortado.dropna()

    print(f"Shape tras recorte: {df_recortado.shape} "
          f"(de {df_features.shape[0]} meses originales)")

    return df_recortado


def evaluar_k_optimo(datos_escalados: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    resultados = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=SEMILLA_ALEATORIA, n_init=10)
        etiquetas = modelo.fit_predict(datos_escalados)
        resultados.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(datos_escalados, etiquetas),
        })

    return pd.DataFrame(resultados)


def entrenar_kmeans_regimen(k: int = K_REGIMENES) -> pd.DataFrame:
    panel = cargar_panel_features_macro()
    datos_clustering = preparar_datos_clustering(panel)

    escalador = StandardScaler()
    datos_escalados = escalador.fit_transform(datos_clustering)

    print("\n=== Evaluación de k (validación de k=4 frente a la literatura) ===")
    evaluacion = evaluar_k_optimo(datos_escalados)
    print(evaluacion.to_string(index=False))

    modelo = KMeans(n_clusters=k, random_state=SEMILLA_ALEATORIA, n_init=10)
    etiquetas = modelo.fit_predict(datos_escalados)

    datos_clustering["regimen"] = etiquetas

    print(f"\n=== Distribución de observaciones por régimen (k={k}) ===")
    print(datos_clustering["regimen"].value_counts().sort_index())

    print("\n=== Centroides (en escala original, deshaciendo estandarización) ===")
    centroides_desescalados = escalador.inverse_transform(modelo.cluster_centers_)
    df_centroides = pd.DataFrame(centroides_desescalados, columns=FEATURES_REGIMEN)

    # Deshace también el log1p para leer el centroide en unidades originales.
    for columna in FEATURES_LOG1P:
        df_centroides[columna] = np.expm1(df_centroides[columna])

    df_centroides.index.name = "regimen"
    print(df_centroides.round(3).to_string())

    panel_con_regimen = panel.loc[datos_clustering.index].copy()
    panel_con_regimen["regimen"] = etiquetas

    return panel_con_regimen


def guardar_panel_con_regimen() -> None:
    panel = entrenar_kmeans_regimen()
    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_con_regimen.csv"
    panel.to_csv(ruta_salida)
    print(f"\nGuardado en {ruta_salida}")


if __name__ == "__main__":
    guardar_panel_con_regimen()