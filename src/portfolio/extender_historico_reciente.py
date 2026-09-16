"""Extiende el histórico de cartera desde el último rebalanceo del walk-forward hasta el más reciente (2025-01-31, 2026-01-31), con el modelo de producción ya entrenado y calendario anual en enero. Tras ejecutarlo, volver a correr calcular_cartera_vigente.py."""

import json

import pandas as pd

from api.main import cargar_historico_predicciones, cargar_modelo_mas_reciente
from config.series import RUTA_DATOS_PROCESADOS
from data_ingestion.universe import get_universe
from factor_prediction.entrenamiento import cargar_panel_maestro, preparar_features
from portfolio.cartera_acciones import (
    N_ACCIONES_MOMENTUM,
    cargar_precios_ticker,
    calcular_retorno_cartera_acciones,
    calcular_turnover,
    construir_cartera_objetivo,
    seleccionar_top_momentum,
)
from portfolio.pesos_cartera import calcular_pesos_cartera

RUTA_HISTORICO_TURNOVER_WF = f"{RUTA_DATOS_PROCESADOS}/historico_turnover_cartera_walk_forward.csv"
RUTA_RETORNO_WF = f"{RUTA_DATOS_PROCESADOS}/retorno_cartera_acciones_walk_forward.csv"
RUTA_ESTADO_VIGENTE = f"{RUTA_DATOS_PROCESADOS}/cartera_vigente_estado.json"

RUTA_SALIDA_TURNOVER_COMPLETO = f"{RUTA_DATOS_PROCESADOS}/historico_turnover_cartera_completo.csv"
RUTA_SALIDA_RETORNO_COMPLETO = f"{RUTA_DATOS_PROCESADOS}/retorno_cartera_acciones_completo.csv"

FECHAS_REBALANCEO_FALTANTES = [pd.Timestamp("2025-01-31"), pd.Timestamp("2026-01-31")]


def _predecir_momentum_en_fecha(fecha: pd.Timestamp) -> float:
    # Mismo mecanismo que calcular_prediccion_vigente() en api/main.py, aplicado
    # a 'fecha' en vez de a la fila más reciente del panel.
    panel = cargar_panel_maestro()
    features = preparar_features(panel)
    fila = features[features["fecha_mes"] == fecha]

    if fila.empty:
        raise ValueError(f"No hay fila en el panel maestro para {fecha.date()}.")

    X = fila.drop(columns=["fecha_mes"])
    modelo = cargar_modelo_mas_reciente()
    historico_columnas = cargar_historico_predicciones().columns

    y_pred = modelo.predict(X)
    prediccion_por_factor = pd.Series(y_pred[0], index=historico_columnas)
    return float(prediccion_por_factor["Mom"])


def _obtener_peso_momentum_en_fechas(fechas: list[pd.Timestamp]) -> pd.Series:
    historico = cargar_historico_predicciones()["Mom"].copy()

    for fecha in fechas:
        if fecha not in historico.index:
            prediccion = _predecir_momentum_en_fecha(fecha)
            historico.loc[fecha] = prediccion
            print(f"  {fecha.date()}: predicción nueva (fuera del test guardado) = {prediccion:.4f}")
        else:
            print(f"  {fecha.date()}: reutilizada de predicciones_test_largo.csv = {historico.loc[fecha]:.4f}")

    historico = historico.sort_index()
    pesos = calcular_pesos_cartera(historico)

    return pesos.loc[fechas, "Mom"]


def _cargar_ultima_cartera_walk_forward() -> tuple[pd.Timestamp, pd.Series]:
    historico = pd.read_csv(RUTA_HISTORICO_TURNOVER_WF, parse_dates=["fecha_rebalanceo"])
    fecha_ultima = historico["fecha_rebalanceo"].max()
    cartera = (
        historico[historico["fecha_rebalanceo"] == fecha_ultima]
        .set_index("ticker")["peso_nuevo"]
    )
    return fecha_ultima, cartera[cartera != 0.0]


def main() -> None:
    print("=== 1. Cargando última cartera conocida del walk-forward ===")
    fecha_anterior, cartera_anterior = _cargar_ultima_cartera_walk_forward()
    print(f"Último rebalanceo del walk-forward: {fecha_anterior.date()}, {len(cartera_anterior)} posiciones")

    print("\n=== 2. Obteniendo peso de Momentum para los rebalanceos faltantes ===")
    pesos_momentum = _obtener_peso_momentum_en_fechas(FECHAS_REBALANCEO_FALTANTES)
    print(pesos_momentum)

    print("\n=== 3. Cargando precios y universos point-in-time ===")
    precios = cargar_precios_ticker(incluir_spy=True)
    universos = {
        fecha: get_universe(fecha, verbose=False)["ticker"].tolist()
        for fecha in FECHAS_REBALANCEO_FALTANTES
    }

    print("\n=== 4. Construyendo las carteras de los rebalanceos faltantes ===")
    carteras_nuevas = {}
    cartera_previa = cartera_anterior
    for fecha in FECHAS_REBALANCEO_FALTANTES:
        seleccion = seleccionar_top_momentum(
            precios, fecha, universo=universos[fecha], n=N_ACCIONES_MOMENTUM
        )
        cartera = construir_cartera_objetivo(seleccion, pesos_momentum.loc[fecha])
        turnover = calcular_turnover(cartera, cartera_previa)
        carteras_nuevas[fecha] = {"cartera": cartera, "turnover": turnover}
        cartera_previa = cartera
        print(f"  {fecha.date()}: {len(cartera) - 1} acciones + SPY, "
              f"{(turnover['accion'] != 'mantener').sum()} operaciones respecto al rebalanceo anterior")

    print("\n=== 5. Calculando retornos mensuales de los folds nuevos ===")
    retorno_nuevo = calcular_retorno_cartera_acciones(carteras_nuevas, precios, meses_por_fold=12)
    print(f"Meses con retorno calculado: {len(retorno_nuevo)} "
          f"({retorno_nuevo.index.min().date()} a {retorno_nuevo.index.max().date()})")

    print("\n=== 6. Concatenando con el retorno del walk-forward histórico ===")
    retorno_wf = pd.read_csv(RUTA_RETORNO_WF, index_col=0, parse_dates=True).iloc[:, 0]
    retorno_completo = pd.concat([retorno_wf, retorno_nuevo]).sort_index()
    retorno_completo = retorno_completo[~retorno_completo.index.duplicated(keep="last")]
    retorno_completo.to_csv(RUTA_SALIDA_RETORNO_COMPLETO, header=["retorno_mensual"])
    print(f"Guardado en {RUTA_SALIDA_RETORNO_COMPLETO} -- "
          f"{retorno_completo.index.min().date()} a {retorno_completo.index.max().date()}, "
          f"{len(retorno_completo)} meses")

    print("\n=== 7. Extendiendo el histórico de turnover (para Power BI) ===")
    historico_turnover_wf = pd.read_csv(RUTA_HISTORICO_TURNOVER_WF, parse_dates=["fecha_rebalanceo"])
    piezas_nuevas = []
    for fecha, contenido in carteras_nuevas.items():
        t = contenido["turnover"].copy()
        t.index.name = "ticker"
        t = t.reset_index()
        t.insert(0, "fecha_rebalanceo", fecha)
        piezas_nuevas.append(t)
    historico_completo = pd.concat([historico_turnover_wf, *piezas_nuevas], ignore_index=True)
    historico_completo.to_csv(RUTA_SALIDA_TURNOVER_COMPLETO, index=False)
    print(f"Guardado en {RUTA_SALIDA_TURNOVER_COMPLETO} -- "
          f"{historico_completo['fecha_rebalanceo'].nunique()} rebalanceos en total")

    print("\n=== 8. Reseteando cartera_vigente_estado.json al calendario correcto ===")
    fecha_estado_correcta = FECHAS_REBALANCEO_FALTANTES[-1]
    cartera_estado_correcta = carteras_nuevas[fecha_estado_correcta]["cartera"]
    estado = {
        "fecha_ultimo_rebalanceo": fecha_estado_correcta.strftime("%Y-%m-%d"),
        "cartera": cartera_estado_correcta.to_dict(),
    }
    with open(RUTA_ESTADO_VIGENTE, "w") as f:
        json.dump(estado, f, indent=2)
    print(f"Estado reseteado: último rebalanceo = {fecha_estado_correcta.date()} "
          f"({len(cartera_estado_correcta)} posiciones)")


if __name__ == "__main__":
    main()