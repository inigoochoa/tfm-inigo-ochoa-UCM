"""Evalúa los modelos de Capa 2: significancia estadística del acierto direccional (test binomial vs p=0.5) y comparación frente a baselines naive ('cero', 'media_train')."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from factor_prediction.entrenamiento import (
    HORIZONTES,
    N_ARBOLES_FINAL,
    SEMILLA_ALEATORIA,
    buscar_mejores_hiperparametros,
    cargar_panel_maestro,
    cargar_target,
    construir_dataset_entrenamiento,
    dividir_train_test,
    preparar_features,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

NIVEL_SIGNIFICANCIA = 0.05


def calcular_acierto_direccional(y_real: pd.Series, y_pred: pd.Series) -> float:
    return (np.sign(y_real) == np.sign(y_pred)).mean()


def test_significancia_direccional(acierto: float, n: int) -> float:
    # Test binomial exacto (no aproximación normal): más apropiado con el tamaño
    # de muestra pequeño del test (65-76 obs).
    n_aciertos = round(acierto * n)
    resultado = binomtest(k=n_aciertos, n=n, p=0.5, alternative="two-sided")
    return resultado.pvalue


def evaluar_baseline_cero(y_test: pd.DataFrame) -> pd.DataFrame:
    # Sin acierto direccional definido (sign(0)=0): se reporta solo R2 y MAE.
    resultados = []
    for columna in y_test.columns:
        y_pred_constante = pd.Series(0.0, index=y_test.index)
        r2 = r2_score(y_test[columna], y_pred_constante)
        mae = mean_absolute_error(y_test[columna], y_pred_constante)
        resultados.append({
            "factor": columna.replace("target_", ""),
            "modelo": "baseline_cero",
            "r2": r2,
            "mae": mae,
            "acierto_direccional": np.nan,
            "p_valor_binomial": np.nan,
        })
    return pd.DataFrame(resultados)


def evaluar_baseline_media_train(
    y_train: pd.DataFrame, y_test: pd.DataFrame
) -> pd.DataFrame:
    resultados = []
    for columna in y_test.columns:
        media_train = y_train[columna].mean()
        y_pred_constante = pd.Series(media_train, index=y_test.index)

        r2 = r2_score(y_test[columna], y_pred_constante)
        mae = mean_absolute_error(y_test[columna], y_pred_constante)
        acierto = calcular_acierto_direccional(y_test[columna], y_pred_constante)
        p_valor = test_significancia_direccional(acierto, len(y_test))

        resultados.append({
            "factor": columna.replace("target_", ""),
            "modelo": "baseline_media_train",
            "r2": r2,
            "mae": mae,
            "acierto_direccional": acierto,
            "p_valor_binomial": p_valor,
        })
    return pd.DataFrame(resultados)


def evaluar_modelo_rf_con_significancia(
    modelo: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> pd.DataFrame:
    y_pred = modelo.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

    resultados = []
    for columna in y_test.columns:
        r2 = r2_score(y_test[columna], y_pred[columna])
        mae = mean_absolute_error(y_test[columna], y_pred[columna])
        acierto = calcular_acierto_direccional(y_test[columna], y_pred[columna])
        p_valor = test_significancia_direccional(acierto, len(y_test))

        resultados.append({
            "factor": columna.replace("target_", ""),
            "modelo": "rf_regularizado",
            "r2": r2,
            "mae": mae,
            "acierto_direccional": acierto,
            "p_valor_binomial": p_valor,
        })
    return pd.DataFrame(resultados)


def evaluar_horizonte_completo(horizonte: str) -> pd.DataFrame:
    panel = cargar_panel_maestro()
    features = preparar_features(panel)
    target = cargar_target(horizonte)

    dataset = construir_dataset_entrenamiento(features, target)
    X_train, X_test, y_train, y_test = dividir_train_test(dataset)

    busqueda = buscar_mejores_hiperparametros(X_train, y_train)
    modelo = RandomForestRegressor(
        n_estimators=N_ARBOLES_FINAL,
        random_state=SEMILLA_ALEATORIA,
        n_jobs=-1,
        **busqueda.best_params_,
    )
    modelo.fit(X_train, y_train)

    resultados_rf = evaluar_modelo_rf_con_significancia(modelo, X_test, y_test)
    resultados_cero = evaluar_baseline_cero(y_test)
    resultados_media = evaluar_baseline_media_train(y_train, y_test)

    resultados = pd.concat(
        [resultados_rf, resultados_media, resultados_cero], ignore_index=True
    )
    resultados["horizonte"] = horizonte
    resultados["significativo_5pct"] = resultados["p_valor_binomial"] < NIVEL_SIGNIFICANCIA

    return resultados


if __name__ == "__main__":
    todos_resultados = []

    for horizonte in HORIZONTES:
        print(f"\n{'=' * 70}")
        print(f"HORIZONTE: {horizonte.upper()}")
        print("=" * 70)
        resultados = evaluar_horizonte_completo(horizonte)
        todos_resultados.append(resultados)

        tabla = resultados.pivot_table(
            index="factor",
            columns="modelo",
            values=["r2", "acierto_direccional"],
        )
        print(f"\n=== Comparación RF vs baselines ({horizonte}) ===")
        print(tabla.round(4).to_string())

        print(f"\n=== Significancia del acierto direccional (p<{NIVEL_SIGNIFICANCIA}) ===")
        cols_significancia = ["factor", "modelo", "acierto_direccional", "p_valor_binomial", "significativo_5pct"]
        print(
            resultados[resultados["modelo"] != "baseline_cero"][cols_significancia]
            .sort_values(["factor", "modelo"])
            .to_string(index=False)
        )

    df_final = pd.concat(todos_resultados, ignore_index=True)
    ruta_salida = "data/processed/evaluacion_capa2_vs_baselines.csv"
    df_final.to_csv(ruta_salida, index=False)
    print(f"\n\nGuardado resumen completo en {ruta_salida}")