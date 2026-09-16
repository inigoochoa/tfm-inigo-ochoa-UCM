"""Random Forest que replica las etiquetas de régimen de k-means, para poder clasificar un mes nuevo sin re-entrenar el clustering. Incluye evaluación e interpretación SHAP."""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config.series import RUTA_DATOS_PROCESADOS
from regime.kmeans_regimen import FEATURES_REGIMEN, FEATURES_LOG1P

FECHA_CORTE_TRAIN = "2019-12-31"

NOMBRES_REGIMEN = {
    0: "Expansión",
    1: "Expansión tardía / Endurecimiento",
    2: "Crisis aguda",
    3: "Recuperación",
}

SEMILLA_ALEATORIA = 42
N_ARBOLES = 500


def cargar_panel_con_regimen() -> pd.DataFrame:
    ruta = f"{RUTA_DATOS_PROCESADOS}/panel_mensual_con_regimen.csv"
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)


def preparar_datos_rf(panel: pd.DataFrame) -> pd.DataFrame:
    # A diferencia de kmeans_regimen.py, no se estandariza: el RF (basado en
    # particiones por umbral, no en distancias) es invariante a la escala.
    df = panel[FEATURES_REGIMEN + ["regimen"]].copy()

    for columna in FEATURES_LOG1P:
        df[columna] = np.log1p(df[columna])

    df = df.dropna()

    return df


def dividir_train_test(df: pd.DataFrame, fecha_corte: str = FECHA_CORTE_TRAIN):
    train = df.loc[:fecha_corte]
    test = df.loc[fecha_corte:].iloc[1:]  # excluir la fecha de corte, ya en train

    X_train, y_train = train[FEATURES_REGIMEN], train["regimen"]
    X_test, y_test = test[FEATURES_REGIMEN], test["regimen"]

    print(f"Train: {X_train.shape[0]} obs ({train.index.min().date()} a {train.index.max().date()})")
    print(f"Test:  {X_test.shape[0]} obs ({test.index.min().date()} a {test.index.max().date()})")
    print(f"\nDistribución de régimen en train:\n{y_train.value_counts().sort_index()}")
    print(f"\nDistribución de régimen en test:\n{y_test.value_counts().sort_index()}")

    return X_train, X_test, y_train, y_test


def entrenar_rf_regimen(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    # class_weight="balanced" compensa el desbalance de "Crisis aguda" (2.9% de las
    # observaciones); se probó "balanced_subsample" y no mejoró el recall.
    modelo = RandomForestClassifier(
        n_estimators=N_ARBOLES,
        class_weight="balanced",
        random_state=SEMILLA_ALEATORIA,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = modelo.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"\n=== F1 macro (test): {f1_macro:.4f} ===")

    print("\n=== Classification report (test) ===")
    etiquetas_presentes = sorted(set(y_test) | set(y_pred))
    nombres_presentes = [NOMBRES_REGIMEN[r] for r in etiquetas_presentes]
    print(classification_report(y_test, y_pred, labels=etiquetas_presentes, target_names=nombres_presentes, zero_division=0))

    print("=== Matriz de confusión (filas=real, columnas=predicho) ===")
    matriz = confusion_matrix(y_test, y_pred, labels=etiquetas_presentes)
    df_matriz = pd.DataFrame(matriz, index=nombres_presentes, columns=nombres_presentes)
    print(df_matriz.to_string())


def entrenar_y_evaluar_rf_regimen() -> RandomForestClassifier:
    panel = cargar_panel_con_regimen()
    df = preparar_datos_rf(panel)

    X_train, X_test, y_train, y_test = dividir_train_test(df)

    modelo = entrenar_rf_regimen(X_train, y_train)
    evaluar_modelo(modelo, X_test, y_test)

    return modelo


def calcular_valores_shap(modelo: RandomForestClassifier, X: pd.DataFrame):
    explicador = shap.TreeExplainer(modelo)
    valores_shap = explicador(X)
    return valores_shap


def resumen_importancia_global(valores_shap, X: pd.DataFrame) -> pd.DataFrame:
    # valores_shap.values tiene forma (n_obs, n_features, n_clases)
    importancia_por_clase = np.abs(valores_shap.values).mean(axis=0)  # (n_features, n_clases)
    importancia_global = importancia_por_clase.mean(axis=1)  # media entre las 4 clases

    df_importancia = pd.DataFrame({
        "feature": X.columns,
        "importancia_shap": importancia_global,
    }).sort_values("importancia_shap", ascending=False).reset_index(drop=True)

    return df_importancia


def graficar_resumen_shap(valores_shap, X: pd.DataFrame, nombre_clase: str, indice_clase: int) -> None:
    import matplotlib.pyplot as plt

    shap.summary_plot(
        valores_shap.values[:, :, indice_clase],
        X,
        show=False,
    )
    plt.title(f"SHAP summary — régimen: {nombre_clase}")
    plt.tight_layout()
    ruta_salida = f"data/processed/shap_summary_{nombre_clase.replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {ruta_salida}")


def analizar_shap_completo(modelo: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    print("Calculando valores SHAP (puede tardar unos segundos)...")
    valores_shap = calcular_valores_shap(modelo, X)

    df_importancia = resumen_importancia_global(valores_shap, X)
    print("\n=== Importancia global de features (media |SHAP|, entre las 4 clases) ===")
    print(df_importancia.to_string(index=False))

    print("\nGenerando summary plots por régimen...")
    for indice_clase, nombre_clase in NOMBRES_REGIMEN.items():
        graficar_resumen_shap(valores_shap, X, nombre_clase, indice_clase)

    return df_importancia


if __name__ == "__main__":
    modelo = entrenar_y_evaluar_rf_regimen()

    panel = cargar_panel_con_regimen()
    df = preparar_datos_rf(panel)
    X_train, X_test, y_train, y_test = dividir_train_test(df)

    print("\n" + "=" * 60)
    print("ANÁLISIS SHAP (sobre conjunto de entrenamiento)")
    print("=" * 60)
    analizar_shap_completo(modelo, X_train)