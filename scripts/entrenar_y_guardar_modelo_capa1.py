"""Entrena el RF de régimen de Capa 1 y lo persiste en models/, ya que
regime/rf_clasificador_regimen.py no lo hace por sí solo (solo se usa para
evaluación/SHAP en el notebook, donde se reentrena en caliente)."""

import joblib

from regime.rf_clasificador_regimen import entrenar_y_evaluar_rf_regimen

RUTA_SALIDA = "models/rf_regimen_capa1.joblib"

if __name__ == "__main__":
    modelo = entrenar_y_evaluar_rf_regimen()
    joblib.dump(modelo, RUTA_SALIDA)
    print(f"\nModelo guardado en: {RUTA_SALIDA}")
