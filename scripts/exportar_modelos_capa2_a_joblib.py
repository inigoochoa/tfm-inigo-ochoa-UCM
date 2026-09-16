"""Extrae los 3 RF finales de Capa 2 desde el tracking de MLflow local y los
persiste en models/ como joblib planos, para que el tutor pueda cargarlos sin
depender del historial de MLflow de esta máquina (que no viaja con el repo)."""

import joblib
import mlflow

NOMBRE_EXPERIMENTO_MLFLOW = "capa2_prediccion_factores"

MODELOS = [
    ("rf_corto_final", "modelo_rf_corto", "models/rf_corto_final.joblib"),
    ("rf_medio_final", "modelo_rf_medio", "models/rf_medio_final.joblib"),
    ("rf_largo_final", "modelo_rf_largo", "models/rf_largo_final.joblib"),
]


def exportar_modelo(nombre_run: str, nombre_artefacto: str, ruta_salida: str) -> None:
    cliente = mlflow.tracking.MlflowClient()
    experimento = cliente.get_experiment_by_name(NOMBRE_EXPERIMENTO_MLFLOW)
    if experimento is None:
        raise RuntimeError(f"No se encontró el experimento '{NOMBRE_EXPERIMENTO_MLFLOW}'.")

    runs = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string=f"tags.mlflow.runName = '{nombre_run}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No se encontró ningún run llamado '{nombre_run}'.")

    run_id = runs[0].info.run_id
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/{nombre_artefacto}")
    joblib.dump(modelo, ruta_salida)
    print(f"{nombre_run} (run {run_id}) -> {ruta_salida}")


if __name__ == "__main__":
    for nombre_run, nombre_artefacto, ruta_salida in MODELOS:
        exportar_modelo(nombre_run, nombre_artefacto, ruta_salida)
