"""HMM gaussiano como benchmark probabilístico del clustering k-means de régimen: modela la dinámica temporal entre estados vía matriz de transición."""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from regime.kmeans_regimen import FEATURES_LOG1P, FEATURES_REGIMEN, RUTA_DATOS_PROCESADOS

N_ESTADOS_HMM = 4
SEMILLA_ALEATORIA = 42
N_ITERACIONES_MAX = 200


def cargar_panel_con_regimen() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_con_regimen.csv"
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)


def preparar_datos_hmm(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # Mismas transformaciones que k-means (log1p + estandarizado): el HMM
    # gaussiano también depende de la escala de las features.
    df_features = panel[FEATURES_REGIMEN].copy()

    for columna in FEATURES_LOG1P:
        df_features[columna] = np.log1p(df_features[columna])

    escalador = StandardScaler()
    datos_escalados = escalador.fit_transform(df_features)

    return df_features, datos_escalados


def entrenar_hmm(datos_escalados: np.ndarray, n_estados: int = N_ESTADOS_HMM) -> GaussianHMM:
    # covariance_type="diag": con 18 features y ~380 observaciones repartidas en 4
    # estados, una covarianza completa sobreajustaría; la diagonal es la opción
    # estándar cuando el número de features es alto relativo a las observaciones.
    modelo = GaussianHMM(
        n_components=n_estados,
        covariance_type="diag",
        n_iter=N_ITERACIONES_MAX,
        random_state=SEMILLA_ALEATORIA,
    )
    modelo.fit(datos_escalados)
    return modelo


def evaluar_hmm_vs_kmeans() -> None:
    panel = cargar_panel_con_regimen()
    df_features, datos_escalados = preparar_datos_hmm(panel)
    regimen_kmeans = panel["regimen"].values

    print(f"Entrenando HMM ({N_ESTADOS_HMM} estados, covarianza diagonal) "
          f"sobre {datos_escalados.shape[0]} observaciones...")
    modelo_hmm = entrenar_hmm(datos_escalados)
    estados_hmm = modelo_hmm.predict(datos_escalados)

    print(f"Log-verosimilitud del modelo: {modelo_hmm.score(datos_escalados):.2f}")

    # Emparejamiento de estados: centroide de cada estado HMM vs.
    # centroide de cada régimen k-means, en el espacio estandarizado.
    centroides_hmm = np.array([
        datos_escalados[estados_hmm == e].mean(axis=0) for e in range(N_ESTADOS_HMM)
    ])
    centroides_kmeans = np.array([
        datos_escalados[regimen_kmeans == r].mean(axis=0) for r in range(N_ESTADOS_HMM)
    ])

    matriz_distancias = np.linalg.norm(
        centroides_hmm[:, None, :] - centroides_kmeans[None, :, :], axis=2
    )
    filas_hmm, columnas_kmeans = linear_sum_assignment(matriz_distancias)
    mapeo_hmm_a_kmeans = dict(zip(filas_hmm, columnas_kmeans))

    print(f"\nEmparejamiento de estados (HMM -> k-means): {mapeo_hmm_a_kmeans}")

    estados_hmm_emparejados = np.array([mapeo_hmm_a_kmeans[e] for e in estados_hmm])

    # Métricas invariantes a la etiqueta (no dependen del
    # emparejamiento, pero se reportan igualmente para contexto)
    ari = adjusted_rand_score(regimen_kmeans, estados_hmm)
    nmi = normalized_mutual_info_score(regimen_kmeans, estados_hmm)
    print(f"\nAdjusted Rand Index (HMM vs k-means): {ari:.4f}")
    print(f"Normalized Mutual Information (HMM vs k-means): {nmi:.4f}")

    print("\n=== Matriz de confusión (filas=k-means, columnas=HMM emparejado) ===")
    matriz_confusion = confusion_matrix(regimen_kmeans, estados_hmm_emparejados)
    print(pd.DataFrame(
        matriz_confusion,
        index=[f"kmeans_{r}" for r in range(N_ESTADOS_HMM)],
        columns=[f"hmm_{r}" for r in range(N_ESTADOS_HMM)],
    ).to_string())

    acuerdo_pct = (estados_hmm_emparejados == regimen_kmeans).mean()
    print(f"\n% de meses donde HMM y k-means coinciden (tras emparejar): {acuerdo_pct:.1%}")

    # Foco específico: ¿el HMM detecta mejor la crisis aguda (régimen
    # 2 de k-means) gracias a su modelado de persistencia temporal?
    mascara_crisis_kmeans = regimen_kmeans == 2
    acuerdo_en_crisis = (estados_hmm_emparejados[mascara_crisis_kmeans] == 2).mean()
    print(f"\nDe los {mascara_crisis_kmeans.sum()} meses de 'Crisis aguda' según k-means, "
          f"el HMM coincide en el {acuerdo_en_crisis:.1%}")

    # Guardar resultado para inspección posterior
    df_comparacion = pd.DataFrame({
        "regimen_kmeans": regimen_kmeans,
        "estado_hmm": estados_hmm_emparejados,
        "coincide": estados_hmm_emparejados == regimen_kmeans,
    }, index=panel.index)
    ruta_salida = f"{RUTA_DATOS_PROCESADOS}/comparacion_hmm_kmeans.csv"
    df_comparacion.to_csv(ruta_salida)
    print(f"\nGuardado en {ruta_salida}")


if __name__ == "__main__":
    evaluar_hmm_vs_kmeans()