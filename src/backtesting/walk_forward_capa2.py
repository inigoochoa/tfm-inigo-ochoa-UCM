"""Walk-forward (expanding window) del modelo de Momentum-12m (Capa 2): reentrena anualmente (recalculando hiperparámetros) desde 2013-12-31, y pasa cada predicción por el mismo mecanismo de tilt de pesos_cartera.py."""

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config.series import RUTA_DATOS_PROCESADOS
from factor_prediction.entrenamiento import (
    N_ARBOLES_FINAL,
    SEMILLA_ALEATORIA,
    buscar_mejores_hiperparametros,
    cargar_panel_maestro,
    cargar_target,
    construir_dataset_entrenamiento,
    preparar_features,
)
from portfolio.evaluacion_cartera import (
    MESES_POR_ANIO,
    calcular_metricas,
    calcular_pesos_rebalanceo_anual,
    calcular_retorno_cartera,
)
from portfolio.pesos_cartera import FACTORES, PESO_BASE, calcular_pesos_cartera

HORIZONTE = "largo"
PRIMER_CORTE = "2013-12-31"

NOMBRE_EXPERIMENTO_MLFLOW = "capa2_walk_forward_momentum"
NOMBRE_FICHERO_PREDICCIONES_SALIDA = "predicciones_walk_forward_momentum.csv"
NOMBRE_FICHERO_METADATOS_SALIDA = "walk_forward_metadatos_folds.csv"
NOMBRE_FICHERO_METRICAS_SALIDA = "walk_forward_comparacion_carteras.csv"


def generar_fechas_de_corte(
    dataset: pd.DataFrame, primer_corte: str = PRIMER_CORTE
) -> list[pd.Timestamp]:
    # Avanza de 12 en 12 filas del índice (no por calendario, para evitar
    # desajustes si algún mes falta), hasta dejar al menos 12 meses de test.
    fechas = dataset.index.sort_values()
    posiciones_corte = fechas.get_indexer([pd.Timestamp(primer_corte)])

    if posiciones_corte[0] == -1:
        raise ValueError(
            f"La fecha de corte {primer_corte} no está en el índice del dataset. "
            "Revisa que exista una observación exacta en esa fecha."
        )

    cortes = []
    idx_corte = posiciones_corte[0]
    while idx_corte + MESES_POR_ANIO < len(fechas):
        cortes.append(fechas[idx_corte])
        idx_corte += MESES_POR_ANIO

    return cortes


def entrenar_y_predecir_fold(
    dataset: pd.DataFrame, corte: pd.Timestamp
) -> tuple[pd.Series, dict]:
    columnas_target = [c for c in dataset.columns if c.startswith("target_")]
    columnas_x = [c for c in dataset.columns if c not in columnas_target]

    train = dataset.loc[:corte]
    test = dataset.loc[corte:].iloc[1 : MESES_POR_ANIO + 1]

    X_train, y_train = train[columnas_x], train[columnas_target]
    X_test = test[columnas_x]

    busqueda = buscar_mejores_hiperparametros(X_train, y_train)

    modelo = RandomForestRegressor(
        n_estimators=N_ARBOLES_FINAL,
        random_state=SEMILLA_ALEATORIA,
        n_jobs=-1,
        **busqueda.best_params_,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    columnas_limpias = [c.replace("target_", "") for c in columnas_target]
    predicciones = pd.DataFrame(y_pred, columns=columnas_limpias, index=X_test.index)

    metadatos_fold = {
        "corte": corte,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mejores_hiperparametros": busqueda.best_params_,
        "r2_cv_train": busqueda.best_score_,
    }

    return predicciones["Mom"], metadatos_fold


def ejecutar_walk_forward() -> tuple[pd.Series, pd.DataFrame]:
    panel = cargar_panel_maestro()
    features = preparar_features(panel)
    target = cargar_target(HORIZONTE)
    dataset = construir_dataset_entrenamiento(features, target).set_index("fecha_mes")

    cortes = generar_fechas_de_corte(dataset)
    print(
        f"Walk-forward: {len(cortes)} folds anuales, "
        f"desde {cortes[0].date()} hasta {cortes[-1].date()}"
    )

    predicciones_por_fold = []
    metadatos_por_fold = []

    mlflow.set_experiment(NOMBRE_EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name="walk_forward_momentum_largo"):
        mlflow.log_param("horizonte", HORIZONTE)
        mlflow.log_param("primer_corte", PRIMER_CORTE)
        mlflow.log_param("n_folds", len(cortes))
        mlflow.log_param("recalcula_hiperparametros_por_fold", True)

        for i, corte in enumerate(cortes, start=1):
            print(f"\n--- Fold {i}/{len(cortes)}: train hasta {corte.date()} ---")

            with mlflow.start_run(run_name=f"fold_{i}_{corte.date()}", nested=True):
                pred_momentum, metadatos = entrenar_y_predecir_fold(dataset, corte)
                predicciones_por_fold.append(pred_momentum)
                metadatos_por_fold.append(metadatos)

                print(
                    f"n_train={metadatos['n_train']}, "
                    f"mejores_hiperparametros={metadatos['mejores_hiperparametros']}, "
                    f"r2_cv_train={metadatos['r2_cv_train']:.4f}"
                )

                mlflow.log_param("corte", str(corte.date()))
                mlflow.log_param("n_train", metadatos["n_train"])
                mlflow.log_params(metadatos["mejores_hiperparametros"])
                mlflow.log_metric("r2_cv_train", metadatos["r2_cv_train"])

        predicciones_momentum_wf = pd.concat(predicciones_por_fold).sort_index()
        metadatos_df = pd.DataFrame(metadatos_por_fold)

        panel_indexado = panel.set_index("fecha_mes")
        retornos = panel_indexado[FACTORES].loc[predicciones_momentum_wf.index]

        pesos_mensuales_wf = calcular_pesos_cartera(predicciones_momentum_wf)
        pesos_anuales_wf = calcular_pesos_rebalanceo_anual(pesos_mensuales_wf)
        pesos_equal_weight = pd.DataFrame(
            PESO_BASE, index=pesos_anuales_wf.index, columns=FACTORES
        )

        retorno_wf = calcular_retorno_cartera(pesos_anuales_wf, retornos)
        retorno_equal_weight = calcular_retorno_cartera(pesos_equal_weight, retornos)

        resultados = pd.DataFrame([
            calcular_metricas(retorno_wf, "Walk-forward con tilt (rebalanceo anual)"),
            calcular_metricas(retorno_equal_weight, "Equal-weight puro (mismo periodo walk-forward)"),
        ]).set_index("cartera")

        for nombre_cartera, fila in resultados.iterrows():
            prefijo = "wf" if "tilt" in nombre_cartera else "ew"
            mlflow.log_metric(f"{prefijo}_sharpe", fila["sharpe"])
            mlflow.log_metric(f"{prefijo}_calmar", fila["calmar"])
            mlflow.log_metric(f"{prefijo}_max_drawdown", fila["max_drawdown"])
            mlflow.log_metric(f"{prefijo}_retorno_anualizado", fila["retorno_anualizado"])

    return predicciones_momentum_wf, metadatos_df, resultados


if __name__ == "__main__":
    predicciones_momentum_wf, metadatos_df, resultados = ejecutar_walk_forward()

    ruta_predicciones = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PREDICCIONES_SALIDA}"
    predicciones_momentum_wf.to_csv(ruta_predicciones, header=["prediccion_momentum"])
    print(f"\nPredicciones walk-forward guardadas en: {ruta_predicciones}")

    ruta_metadatos = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_METADATOS_SALIDA}"
    metadatos_df.to_csv(ruta_metadatos, index=False)
    print(f"Metadatos de folds guardados en: {ruta_metadatos}")

    print("\n=== Comparación walk-forward: con tilt vs. equal-weight ===\n")
    print(resultados.round(4).to_string())

    ruta_metricas = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_METRICAS_SALIDA}"
    resultados.to_csv(ruta_metricas)
    print(f"\nResultados guardados en: {ruta_metricas}")