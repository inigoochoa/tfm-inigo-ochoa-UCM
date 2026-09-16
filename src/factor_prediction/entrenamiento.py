"""Entrena los 3 Random Forest multi-output de Capa 2 (uno por horizonte: corto=1m, medio=3m, largo=12m), prediciendo el retorno futuro de los 7 factores objetivo. El régimen se excluye de las features por defecto (ver ablacion_regimen.py)."""

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from config.series import RUTA_DATOS_PROCESADOS

FACTORES_OBJETIVO = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "bab_usa"]
COLUMNAS_EXCLUIDAS_DE_FEATURES = FACTORES_OBJETIVO + ["RF", "regimen", "regimen_predicho"]

HORIZONTES = ["corto", "medio", "largo"]

FECHA_CORTE_TRAIN = "2019-12-31"
SEMILLA_ALEATORIA = 42
N_ARBOLES_BUSQUEDA = 200
N_ARBOLES_FINAL = 500
N_SPLITS_TIME_SERIES = 5

NOMBRE_FICHERO_PANEL_MAESTRO = "panel_maestro_capa2.csv"
NOMBRE_EXPERIMENTO_MLFLOW = "capa2_prediccion_factores"

REJILLA_HIPERPARAMETROS = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_leaf": [5, 10, 20, 30],
    "max_features": ["sqrt", 0.5, 1.0],
}


def cargar_panel_maestro(ruta_directorio: str = RUTA_DATOS_PROCESADOS) -> pd.DataFrame:
    ruta = f"{ruta_directorio}/{NOMBRE_FICHERO_PANEL_MAESTRO}"
    return pd.read_csv(ruta, parse_dates=["fecha_mes"])


def preparar_features(panel: pd.DataFrame, incluir_regimen: bool = False) -> pd.DataFrame:
    # incluir_regimen=True añade 'regimen_predicho' (nunca 'regimen', que tiene leak,
    # ver regimen_sin_leakage.py) como dummies; solo se usa desde ablacion_regimen.py.
    columnas_a_excluir = set(COLUMNAS_EXCLUIDAS_DE_FEATURES)
    columnas_features = [
        c for c in panel.columns
        if c not in columnas_a_excluir and c != "fecha_mes"
    ]

    df_features = panel[["fecha_mes"] + columnas_features].copy()

    if incluir_regimen:
        regimen_dummies = pd.get_dummies(panel["regimen_predicho"], prefix="regimen")
        df_features = pd.concat([df_features, regimen_dummies], axis=1)

    return df_features


def cargar_target(
    horizonte: str, ruta_directorio: str = RUTA_DATOS_PROCESADOS
) -> pd.DataFrame:
    ruta = f"{ruta_directorio}/target_{horizonte}.csv"
    return pd.read_csv(ruta, parse_dates=["fecha_mes"])


def construir_dataset_entrenamiento(
    features: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    dataset = features.merge(target, on="fecha_mes", how="inner")
    n_antes = len(dataset)
    dataset = dataset.dropna().reset_index(drop=True)
    n_despues = len(dataset)

    print(
        f"Dataset construido: {n_despues} filas completas "
        f"(de {n_antes}, {n_antes - n_despues} descartadas por NaN)."
    )

    return dataset


def dividir_train_test(dataset: pd.DataFrame, fecha_corte: str = FECHA_CORTE_TRAIN):
    columnas_target = [c for c in dataset.columns if c.startswith("target_")]
    columnas_x = [
        c for c in dataset.columns if c not in columnas_target and c != "fecha_mes"
    ]

    dataset = dataset.set_index("fecha_mes")
    train = dataset.loc[:fecha_corte]
    test = dataset.loc[fecha_corte:].iloc[1:]

    X_train, y_train = train[columnas_x], train[columnas_target]
    X_test, y_test = test[columnas_x], test[columnas_target]

    print(
        f"Train: {X_train.shape[0]} obs ({train.index.min().date()} a {train.index.max().date()})"
    )
    print(
        f"Test:  {X_test.shape[0]} obs ({test.index.min().date()} a {test.index.max().date()})"
    )

    return X_train, X_test, y_train, y_test


def buscar_mejores_hiperparametros(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> GridSearchCV:
    modelo_base = RandomForestRegressor(
        n_estimators=N_ARBOLES_BUSQUEDA,
        random_state=SEMILLA_ALEATORIA,
        n_jobs=-1,
    )

    validacion_temporal = TimeSeriesSplit(n_splits=N_SPLITS_TIME_SERIES)

    busqueda = GridSearchCV(
        estimator=modelo_base,
        param_grid=REJILLA_HIPERPARAMETROS,
        cv=validacion_temporal,
        scoring="r2",
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)

    print(f"\nMejores hiperparámetros: {busqueda.best_params_}")
    print(f"Mejor R2 medio en validación (TimeSeriesSplit, {N_SPLITS_TIME_SERIES} folds): "
          f"{busqueda.best_score_:.4f}")

    return busqueda


def evaluar_modelo(
    modelo: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> pd.DataFrame:
    y_pred = modelo.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

    resultados = []
    for columna in y_test.columns:
        r2 = r2_score(y_test[columna], y_pred[columna])
        mae = mean_absolute_error(y_test[columna], y_pred[columna])
        acierto_direccional = (
            np.sign(y_test[columna]) == np.sign(y_pred[columna])
        ).mean()

        resultados.append({
            "factor": columna.replace("target_", ""),
            "r2": r2,
            "mae": mae,
            "acierto_direccional": acierto_direccional,
        })

    return pd.DataFrame(resultados)


def guardar_predicciones_test(
    modelo: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    horizonte: str,
    ruta_directorio: str = RUTA_DATOS_PROCESADOS,
) -> pd.DataFrame:
    # Consumido directamente por pesos_cartera.py: quien llama es responsable de no
    # invocar esto con un modelo entrenado con incluir_regimen=True.
    y_pred = modelo.predict(X_test)
    predicciones = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)
    predicciones.columns = [c.replace("target_", "") for c in predicciones.columns]
    predicciones.index.name = "fecha_mes"

    ruta = f"{ruta_directorio}/predicciones_test_{horizonte}.csv"
    predicciones.to_csv(ruta)
    print(f"Predicciones de test guardadas en: {ruta}")

    return predicciones


def entrenar_horizonte(
    horizonte: str,
    incluir_regimen: bool = False,
    nombre_run: str | None = None,
) -> tuple[RandomForestRegressor, pd.DataFrame]:
    # nombre_run por defecto es f"rf_{horizonte}_final", el nombre que api/main.py
    # busca como modelo de producción: cualquier llamada con incluir_regimen=True
    # DEBE pasar un nombre_run distinto para no contaminarlo.
    panel = cargar_panel_maestro()
    features = preparar_features(panel, incluir_regimen=incluir_regimen)
    target = cargar_target(horizonte)

    dataset = construir_dataset_entrenamiento(features, target)
    X_train, X_test, y_train, y_test = dividir_train_test(dataset)

    print(f"\nBuscando hiperparámetros (TimeSeriesSplit, {N_SPLITS_TIME_SERIES} folds)...")
    busqueda = buscar_mejores_hiperparametros(X_train, y_train)

    nombre_run_final = nombre_run or f"rf_{horizonte}_final"

    mlflow.set_experiment(NOMBRE_EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name=nombre_run_final):
        mlflow.log_param("horizonte", horizonte)
        mlflow.log_param("n_estimators_final", N_ARBOLES_FINAL)
        mlflow.log_param("fecha_corte_train", FECHA_CORTE_TRAIN)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("incluye_regimen", incluir_regimen)
        mlflow.log_params(busqueda.best_params_)
        mlflow.log_metric("r2_cv_train", busqueda.best_score_)

        modelo_final = RandomForestRegressor(
            n_estimators=N_ARBOLES_FINAL,
            random_state=SEMILLA_ALEATORIA,
            n_jobs=-1,
            **busqueda.best_params_,
        )
        modelo_final.fit(X_train, y_train)

        resultados = evaluar_modelo(modelo_final, X_test, y_test)
        print(f"\n=== Resultados por factor (horizonte: {horizonte}, modelo final) ===")
        print(resultados.to_string(index=False))

        # Solo se persiste en la configuración de producción, para no sobrescribir
        # con una ejecución de ablación el CSV que usan pesos_cartera.py / api/main.py.
        if not incluir_regimen:
            guardar_predicciones_test(modelo_final, X_test, y_test, horizonte)

        for _, fila in resultados.iterrows():
            mlflow.log_metric(f"r2_{fila['factor']}", fila["r2"])
            mlflow.log_metric(f"mae_{fila['factor']}", fila["mae"])
            mlflow.log_metric(f"acierto_direccional_{fila['factor']}", fila["acierto_direccional"])

        mlflow.log_metric("r2_medio", resultados["r2"].mean())
        mlflow.log_metric("acierto_direccional_medio", resultados["acierto_direccional"].mean())

        mlflow.sklearn.log_model(modelo_final, f"modelo_rf_{horizonte}")

    return modelo_final, resultados


if __name__ == "__main__":
    resultados_por_horizonte = {}

    for horizonte in HORIZONTES:
        print(f"\n{'=' * 60}")
        print(f"HORIZONTE: {horizonte.upper()}")
        print("=" * 60)
        modelo, resultados = entrenar_horizonte(horizonte)
        resultados_por_horizonte[horizonte] = resultados

    print("\n\n=== Resumen: acierto direccional medio por horizonte (modelo final) ===")
    for horizonte, resultados in resultados_por_horizonte.items():
        print(f"{horizonte}: {resultados['acierto_direccional'].mean():.3f}")