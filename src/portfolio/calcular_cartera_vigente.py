"""Calcula la recomendación de compra/venta vigente respetando la cadencia anual: si no han pasado ~12 meses desde el último rebalanceo, mantiene la cartera sin cambios."""

import json
from pathlib import Path

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from data_ingestion.universe import get_universe
from portfolio.cartera_acciones import (
    N_ACCIONES_MOMENTUM,
    NUCLEO_TICKER,
    cargar_precios_ticker,
    calcular_turnover,
    construir_cartera_objetivo,
    seleccionar_top_momentum,
)

MESES_MINIMOS_ENTRE_REBALANCEOS = 12

NOMBRE_FICHERO_HISTORICO_TURNOVER = "historico_turnover_cartera_walk_forward.csv"
NOMBRE_FICHERO_PREDICCION_VIGENTE = "prediccion_vigente.json"
NOMBRE_FICHERO_ESTADO = "cartera_vigente_estado.json"
NOMBRE_FICHERO_TURNOVER_VIGENTE = "turnover_vigente.csv"


def _meses_transcurridos(fecha_inicio: pd.Timestamp, fecha_fin: pd.Timestamp) -> int:
    return (fecha_fin.year - fecha_inicio.year) * 12 + (fecha_fin.month - fecha_inicio.month)


def _cargar_estado() -> dict | None:
    ruta = Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_ESTADO
    if not ruta.exists():
        return None

    with open(ruta) as f:
        estado = json.load(f)

    estado["fecha_ultimo_rebalanceo"] = pd.Timestamp(estado["fecha_ultimo_rebalanceo"])
    estado["cartera"] = pd.Series(estado["cartera"])
    return estado


def _bootstrapear_desde_walk_forward() -> dict:
    # Primera ejecución sin estado persistido: parte del último rebalanceo real del
    # walk-forward, en vez de asumir 100% SPY.
    ruta = Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_HISTORICO_TURNOVER
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta ejecutar_cartera_acciones.py --modo walk_forward "
            "primero (genera el histórico del que arranca la cartera vigente)."
        )

    historico = pd.read_csv(ruta, parse_dates=["fecha_rebalanceo"])
    fecha_ultimo = historico["fecha_rebalanceo"].max()
    ultimo_rebalanceo = historico[historico["fecha_rebalanceo"] == fecha_ultimo]

    cartera = ultimo_rebalanceo.set_index("ticker")["peso_nuevo"]
    cartera = cartera[cartera != 0.0]

    print(
        f"Sin estado previo -- bootstrap desde el walk-forward: "
        f"último rebalanceo real en {fecha_ultimo.date()}, {len(cartera)} posiciones."
    )

    return {"fecha_ultimo_rebalanceo": fecha_ultimo, "cartera": cartera}


def _leer_peso_momentum_vigente() -> float:
    # Lee de la caché de la API en vez de recalcular el tilt aquí, para no duplicar
    # la lógica ya validada de pesos_cartera.py.
    ruta = Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_PREDICCION_VIGENTE
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta POST /recalcular-factores en la API (o "
            "factor_prediction/entrenamiento.py) antes de calcular la "
            "cartera vigente, para tener un peso de Momentum actualizado."
        )

    with open(ruta) as f:
        prediccion = json.load(f)

    return float(prediccion["pesos_cartera"]["Mom"])


def _guardar_estado(fecha_rebalanceo: pd.Timestamp, cartera: pd.Series) -> None:
    ruta = Path(RUTA_DATOS_PROCESADOS) / NOMBRE_FICHERO_ESTADO
    estado = {
        "fecha_ultimo_rebalanceo": fecha_rebalanceo.strftime("%Y-%m-%d"),
        "cartera": cartera.to_dict(),
    }
    with open(ruta, "w") as f:
        json.dump(estado, f, indent=2)
    print(f"Estado actualizado en {ruta}")


def calcular_cartera_vigente() -> pd.DataFrame:
    estado_previo = _cargar_estado() or _bootstrapear_desde_walk_forward()
    fecha_ultimo_rebalanceo = estado_previo["fecha_ultimo_rebalanceo"]
    cartera_anterior = estado_previo["cartera"]

    precios = cargar_precios_ticker()
    fecha_actual = precios.index.max()

    meses_desde_ultimo = _meses_transcurridos(fecha_ultimo_rebalanceo, fecha_actual)
    toca_rebalancear = meses_desde_ultimo >= MESES_MINIMOS_ENTRE_REBALANCEOS

    print(f"Último rebalanceo: {fecha_ultimo_rebalanceo.date()}")
    print(f"Fecha actual (último precio disponible): {fecha_actual.date()}")
    print(f"Meses transcurridos: {meses_desde_ultimo} (mínimo requerido: {MESES_MINIMOS_ENTRE_REBALANCEOS})")

    if not toca_rebalancear:
        print("No toca rebalancear todavía. La cartera se mantiene sin cambios.")
        turnover = calcular_turnover(cartera_anterior, cartera_anterior)
        turnover["fecha_calculo"] = fecha_actual
        turnover["rebalanceo_ejecutado"] = False
        turnover.to_csv(f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_TURNOVER_VIGENTE}")
        return turnover

    print("Toca rebalancear -- calculando nueva selección de acciones...")

    peso_momentum_vigente = _leer_peso_momentum_vigente()
    universo_vigente = get_universe(fecha_actual, verbose=False)["ticker"].tolist()

    seleccion = seleccionar_top_momentum(
        precios, fecha_actual, universo=universo_vigente, n=N_ACCIONES_MOMENTUM
    )
    cartera_nueva = construir_cartera_objetivo(seleccion, peso_momentum_vigente)

    turnover = calcular_turnover(cartera_nueva, cartera_anterior)
    turnover["fecha_calculo"] = fecha_actual
    turnover["rebalanceo_ejecutado"] = True

    turnover.to_csv(f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_TURNOVER_VIGENTE}")
    print(f"Turnover vigente guardado en {RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_TURNOVER_VIGENTE}")

    _guardar_estado(fecha_actual, cartera_nueva)

    return turnover

def _turnover_a_dict(turnover: pd.DataFrame) -> dict:
    fecha_calculo = pd.Timestamp(turnover["fecha_calculo"].iloc[0]).strftime("%Y-%m-%d")
    rebalanceo_ejecutado = bool(turnover["rebalanceo_ejecutado"].iloc[0])

    posiciones_df = turnover.drop(columns=["fecha_calculo", "rebalanceo_ejecutado"]).copy()
    posiciones_df.index.name = "ticker"
    posiciones = posiciones_df.reset_index().to_dict(orient="records")

    return {
        "fecha_calculo": fecha_calculo,
        "rebalanceo_ejecutado": rebalanceo_ejecutado,
        "posiciones": posiciones,
    }


def calcular_y_guardar_cartera_vigente_json() -> dict:
    turnover = calcular_cartera_vigente()
    resultado = _turnover_a_dict(turnover)

    ruta = Path(RUTA_DATOS_PROCESADOS) / "turnover_vigente.json"
    with open(ruta, "w") as f:
        json.dump(resultado, f, indent=2)

    return resultado

if __name__ == "__main__":
    turnover_vigente = calcular_cartera_vigente()
    print("\n=== Turnover vigente ===")
    print(turnover_vigente.to_string())