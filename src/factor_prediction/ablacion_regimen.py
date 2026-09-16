"""Análisis de ablación: compara el modelo de Capa 2 con y sin las dummies de régimen como feature."""

import pandas as pd

from factor_prediction.entrenamiento import HORIZONTES, entrenar_horizonte


def comparar_con_sin_regimen() -> pd.DataFrame:
    filas = []

    for horizonte in HORIZONTES:
        print(f"\n{'=' * 60}")
        print(f"HORIZONTE: {horizonte.upper()} — CON régimen")
        print("=" * 60)
        # nombre_run distinto de f"rf_{horizonte}_final": ese es el que api/main.py
        # busca como modelo de producción, y un run de ablación no debe poder ocuparlo.
        _, resultados_con = entrenar_horizonte(
            horizonte,
            incluir_regimen=True,
            nombre_run=f"rf_{horizonte}_ablacion_con_regimen",
        )
        resultados_con["version"] = "con_regimen"
        resultados_con["horizonte"] = horizonte

        print(f"\n{'=' * 60}")
        print(f"HORIZONTE: {horizonte.upper()} — SIN régimen")
        print("=" * 60)
        _, resultados_sin = entrenar_horizonte(
            horizonte,
            incluir_regimen=False,
            nombre_run=f"rf_{horizonte}_ablacion_sin_regimen",
        )
        resultados_sin["version"] = "sin_regimen"
        resultados_sin["horizonte"] = horizonte

        filas.append(resultados_con)
        filas.append(resultados_sin)

    return pd.concat(filas, ignore_index=True)


if __name__ == "__main__":
    resultados = comparar_con_sin_regimen()

    tabla = resultados.pivot_table(
        index=["horizonte", "factor"],
        columns="version",
        values=["r2", "acierto_direccional"],
    )
    print("\n\n=== Comparación CON vs SIN régimen (todos los horizontes) ===")
    print(tabla.round(4).to_string())

    resultados.to_csv("data/processed/ablacion_regimen.csv", index=False)
    print("\nGuardado en data/processed/ablacion_regimen.csv")