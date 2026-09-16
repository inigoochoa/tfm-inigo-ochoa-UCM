"""API que sirve la cartera (GET, caché precalculada) y la recalcula bajo demanda con inferencia ligera, sin reentrenar (POST, protegido por API key)."""

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

from config.series import RUTA_DATOS_PROCESADOS
from factor_prediction.entrenamiento import cargar_panel_maestro, preparar_features
from portfolio.calcular_cartera_vigente import calcular_y_guardar_cartera_vigente_json
from portfolio.pesos_cartera import (
    FACTORES,
    calcular_pesos_cartera,
    calcular_tilt,
    calcular_zscore_expanding,
)
from regime.rf_clasificador_regimen import NOMBRES_REGIMEN

NOMBRE_EXPERIMENTO_MLFLOW = "capa2_prediccion_factores"
NOMBRE_RUN_MODELO = "rf_largo_final"
NOMBRE_ARTEFACTO_MODELO = "modelo_rf_largo"

NOMBRE_FICHERO_PREDICCIONES_HISTORICAS = "predicciones_test_largo.csv"
NOMBRE_FICHERO_CACHE = "prediccion_vigente.json"
NOMBRE_FICHERO_CACHE_CARTERA = "turnover_vigente.json"

VARIABLE_ENTORNO_API_KEY = "TFM_API_KEY"

app = FastAPI(
    title="TFM Factor Timing — API de cartera",
    description=(
        "Sirve la cartera long-only ponderada por factores del TFM "
        "(régimen de mercado + factor timing, UCM), y la cartera de "
        "acciones (core-satélite) derivada de ella."
    ),
    version="0.2.0",
)


class PesosCartera(BaseModel):
    Mkt_RF: float
    SMB: float
    HML: float
    RMW: float
    CMA: float
    Mom: float
    bab_usa: float

    class Config:
        # "Mkt-RF" no es válido como nombre de atributo Python; se mapea desde Mkt_RF.
        populate_by_name = True
        fields = {"Mkt_RF": "Mkt-RF"}


class RegimenInformativoCapa1(BaseModel):
    etiqueta: str
    fecha_mes: str


class PrediccionResponse(BaseModel):
    fecha_mes: str
    prediccion_momentum_largo: float
    z_score: float | None
    tilt_aplicado: float
    pesos_cartera: dict[str, float]
    # Contexto informativo: régimen vigente según el clasificador de Capa 1.
    # NO es un input del pipeline de features de Capa 2 (ver ablacion_regimen.py:
    # incluirlo como feature no mejoró el modelo y se descartó explícitamente).
    regimen_informativo_capa1: RegimenInformativoCapa1
    generado_en: str


class PosicionCartera(BaseModel):
    ticker: str
    peso_anterior: float
    peso_nuevo: float
    delta: float
    accion: str


class CarteraResponse(BaseModel):
    fecha_calculo: str
    rebalanceo_ejecutado: bool
    posiciones: list[PosicionCartera]


def _ruta_cache() -> Path:
    return Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_CACHE


def _ruta_cache_cartera() -> Path:
    return Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_CACHE_CARTERA


@lru_cache(maxsize=1)
def cargar_modelo_mas_reciente():
    cliente = mlflow.tracking.MlflowClient()
    experimento = cliente.get_experiment_by_name(NOMBRE_EXPERIMENTO_MLFLOW)

    if experimento is None:
        raise RuntimeError(
            f"No se encontró el experimento MLflow '{NOMBRE_EXPERIMENTO_MLFLOW}'. "
            "Ejecuta factor_prediction/entrenamiento.py al menos una vez."
        )

    runs = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string=f"tags.mlflow.runName = '{NOMBRE_RUN_MODELO}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            f"No se encontró ningún run llamado '{NOMBRE_RUN_MODELO}'. "
            "Ejecuta factor_prediction/entrenamiento.py al menos una vez."
        )

    run_id = runs[0].info.run_id
    uri_modelo = f"runs:/{run_id}/{NOMBRE_ARTEFACTO_MODELO}"
    modelo = mlflow.sklearn.load_model(uri_modelo)

    return modelo


def cargar_historico_predicciones() -> pd.DataFrame:
    ruta = Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_PREDICCIONES_HISTORICAS
    return pd.read_csv(ruta, parse_dates=["fecha_mes"], index_col="fecha_mes")


def obtener_regimen_informativo(panel: pd.DataFrame, fecha_vigente) -> dict:
    # Lee 'regimen_predicho', ya calculado en el panel maestro por el RF de Capa 1
    # (ver regime/regimen_sin_leakage.py): no reajusta el clustering ni reentrena nada,
    # es la misma inferencia ligera que se usa en producción para el histórico.
    fila_regimen = panel.loc[panel["fecha_mes"] == fecha_vigente].iloc[0]
    codigo_regimen = int(fila_regimen["regimen_predicho"])
    return {
        "etiqueta": NOMBRES_REGIMEN[codigo_regimen],
        "fecha_mes": fecha_vigente.strftime("%Y-%m-%d"),
    }


def calcular_prediccion_vigente() -> dict:
    panel = cargar_panel_maestro()
    features = preparar_features(panel)
    fila_reciente = features.iloc[[-1]]
    fecha_vigente = fila_reciente["fecha_mes"].iloc[0]

    # IMPORTANTE: este campo es puramente informativo. preparar_features() ya excluye
    # 'regimen_predicho' del pipeline de features de Capa 2 por defecto (incluir_regimen=False,
    # ver factor_prediction/entrenamiento.py) tras la ablación que descartó usarlo como input
    # (ablacion_regimen.py). Se calcula aparte, después de X_vigente, para no tocar ese pipeline.
    regimen_informativo = obtener_regimen_informativo(panel, fecha_vigente)

    X_vigente = fila_reciente.drop(columns=["fecha_mes"])

    modelo = cargar_modelo_mas_reciente()
    historico = cargar_historico_predicciones()

    y_pred = modelo.predict(X_vigente)
    prediccion_por_factor = pd.Series(y_pred[0], index=historico.columns)
    prediccion_momentum_nueva = float(prediccion_por_factor["Mom"])

    serie_momentum_historica = historico["Mom"].copy()
    serie_momentum_extendida = serie_momentum_historica.copy()
    serie_momentum_extendida.loc[fecha_vigente] = prediccion_momentum_nueva
    serie_momentum_extendida = serie_momentum_extendida.sort_index()

    z = calcular_zscore_expanding(serie_momentum_extendida)
    tilt = calcular_tilt(z)
    pesos = calcular_pesos_cartera(serie_momentum_extendida)

    z_vigente = z.loc[fecha_vigente]
    tilt_vigente = tilt.loc[fecha_vigente]
    fila_pesos_vigente = pesos.loc[fecha_vigente]

    return {
        "fecha_mes": fecha_vigente.strftime("%Y-%m-%d"),
        "prediccion_momentum_largo": prediccion_momentum_nueva,
        "z_score": None if pd.isna(z_vigente) else float(z_vigente),
        "tilt_aplicado": float(tilt_vigente),
        "pesos_cartera": {factor: float(fila_pesos_vigente[factor]) for factor in FACTORES},
        "regimen_informativo_capa1": regimen_informativo,
        "generado_en": datetime.now(timezone.utc).isoformat(),
    }


def guardar_en_cache(resultado: dict) -> None:
    with open(_ruta_cache(), "w") as f:
        json.dump(resultado, f, indent=2)


def leer_de_cache() -> dict:
    ruta = _ruta_cache()
    if not ruta.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No hay ninguna predicción precalculada todavía. "
                "Ejecuta POST /recalcular-factores primero."
            ),
        )
    with open(ruta) as f:
        return json.load(f)


def leer_de_cache_cartera() -> dict:
    ruta = _ruta_cache_cartera()
    if not ruta.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No hay ninguna cartera de acciones precalculada todavía. "
                "Ejecuta POST /recalcular-cartera primero."
            ),
        )
    with open(ruta) as f:
        return json.load(f)


def verificar_api_key(x_api_key: str = Header(...)) -> None:
    api_key_esperada = os.getenv(VARIABLE_ENTORNO_API_KEY)

    if not api_key_esperada:
        raise HTTPException(
            status_code=500,
            detail=f"La variable de entorno {VARIABLE_ENTORNO_API_KEY} no está configurada en el servidor.",
        )

    if x_api_key != api_key_esperada:
        raise HTTPException(status_code=401, detail="API key inválida.")


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.get("/prediccion-factores", response_model=PrediccionResponse)
def obtener_prediccion():
    """Sirve la última predicción/cartera precalculada (lectura de fichero, sin ejecutar el modelo).

    Incluye 'regimen_informativo_capa1' como contexto de mercado (régimen vigente
    según el clasificador de Capa 1). Es puramente informativo: no forma parte del
    pipeline de features que alimenta la predicción de Capa 2 (ver ablacion_regimen.py).
    """
    return leer_de_cache()


@app.post("/recalcular-factores", response_model=PrediccionResponse)
def recalcular(x_api_key: str = Header(...)):
    """Ejecuta la inferencia ligera sobre los datos más recientes y actualiza la caché de GET /prediccion-factores."""
    verificar_api_key(x_api_key)
    resultado = calcular_prediccion_vigente()
    guardar_en_cache(resultado)
    return resultado


@app.get("/cartera-acciones", response_model=CarteraResponse)
def obtener_cartera():
    """Sirve la última cartera de acciones precalculada (lectura de fichero)."""
    return leer_de_cache_cartera()


@app.post("/recalcular-cartera", response_model=CarteraResponse)
def recalcular_cartera(x_api_key: str = Header(...)):
    # Requiere que POST /recalcular-factores se haya ejecutado antes en este ciclo: lee el
    # peso de Momentum desde prediccion_vigente.json, no lo recalcula. Endpoint aparte de
    # /recalcular-factores porque carga el histórico completo de precios (más pesado).
    verificar_api_key(x_api_key)
    return calcular_y_guardar_cartera_vigente_json()