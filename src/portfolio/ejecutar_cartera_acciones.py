"""Orquesta el backtest de la cartera de acciones core-satélite. --modo selecciona la fuente de señal y cadencia de rebalanceo: simple (pesos ya validados, anual), walk_forward (predicciones walk-forward, anual, modo de producción), trimestral (walk_forward con reconstitución cada 3 meses)."""

import argparse

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from portfolio.cartera_acciones import (
    N_ACCIONES_MOMENTUM,
    NUCLEO_TICKER,
    cargar_precios_ticker,
    calcular_retorno_cartera_acciones,
    construir_serie_carteras,
    construir_universos_por_fecha,
    exportar_historico_turnover,
)
from portfolio.evaluacion_cartera import calcular_metricas, calcular_pesos_rebalanceo_anual
from portfolio.pesos_cartera import calcular_pesos_cartera

NOMBRE_FICHERO_PESOS_FACTOR = "pesos_cartera.csv"
NOMBRE_FICHERO_PREDICCIONES_WF = "predicciones_walk_forward_momentum.csv"
NOMBRE_FICHERO_HISTORICO_TURNOVER = "historico_turnover_cartera_walk_forward.csv"
MESES_POR_FOLD_TRIMESTRAL = 3

SUFIJO_SALIDA_POR_MODO = {
    "simple": "",
    "walk_forward": "_walk_forward",
    "trimestral": "_trimestral",
}


def _cargar_pesos_momentum_anuales(modo: str) -> pd.Series:
    if modo == "simple":
        pesos_mensuales = pd.read_csv(
            f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PESOS_FACTOR}",
            parse_dates=["fecha_mes"],
            index_col="fecha_mes",
        )
    else:
        predicciones_wf = pd.read_csv(
            f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_PREDICCIONES_WF}",
            parse_dates=["fecha_mes"],
            index_col="fecha_mes",
        )
        pesos_mensuales = calcular_pesos_cartera(predicciones_wf["prediccion_momentum"])

    meses_por_fold = MESES_POR_FOLD_TRIMESTRAL if modo == "trimestral" else 12
    fechas_rebalanceo = pesos_mensuales.index[::meses_por_fold]

    if modo == "trimestral":
        # calcular_pesos_rebalanceo_anual() fuerza cadencia de 12 meses; incompatible aquí.
        pesos_en_rebalanceo = pesos_mensuales
    else:
        pesos_en_rebalanceo = calcular_pesos_rebalanceo_anual(pesos_mensuales)

    return pesos_en_rebalanceo.loc[fechas_rebalanceo, "Mom"]


def ejecutar(modo: str) -> None:
    sufijo = SUFIJO_SALIDA_POR_MODO[modo]
    meses_por_fold = MESES_POR_FOLD_TRIMESTRAL if modo == "trimestral" else 12

    print(f"=== 1. Cargando señal de Momentum (modo={modo}) ===")
    pesos_momentum = _cargar_pesos_momentum_anuales(modo)
    fechas_rebalanceo = pesos_momentum.index
    print(f"Fechas de rebalanceo: {len(fechas_rebalanceo)}")
    print(pesos_momentum)

    print("\n=== 2. Cargando precios por ticker (Sharadar delistados + Yahoo activos, incluye SPY) ===")
    precios = cargar_precios_ticker(incluir_spy=True)
    print(f"Shape precios: {precios.shape}")

    print("\n=== 3. Construyendo universos point-in-time por fecha de rebalanceo ===")
    universos = construir_universos_por_fecha(fechas_rebalanceo)
    for fecha, universo in universos.items():
        print(f"  {fecha.date()}: {len(universo)} tickers elegibles")

    print(f"\n=== 4. Construyendo carteras con top {N_ACCIONES_MOMENTUM} acciones de Momentum ===")
    carteras = construir_serie_carteras(
        precios=precios,
        pesos_momentum_anuales=pesos_momentum,
        universos_por_fecha=universos,
    )

    print("\n=== 5. Calculando retorno mensual de la cartera de acciones ===")
    retorno_cartera_acciones = calcular_retorno_cartera_acciones(
        carteras, precios, meses_por_fold=meses_por_fold
    )
    print(f"Meses con retorno calculado: {len(retorno_cartera_acciones)}")
    print(retorno_cartera_acciones.describe())

    print("\n=== 6. Comparando métricas: cartera core-satélite vs. SPY puro ===")
    retorno_spy = precios[NUCLEO_TICKER].pct_change().loc[retorno_cartera_acciones.index]

    resultados = pd.DataFrame([
        calcular_metricas(retorno_cartera_acciones, f"Cartera core-satélite ({modo})"),
        calcular_metricas(retorno_spy, "SPY puro (buy & hold)"),
    ]).set_index("cartera")

    print(resultados.round(4).to_string())

    if modo == "trimestral":
        n_rebalanceos = len(fechas_rebalanceo)
        n_operaciones_totales = sum(
            (carteras[f]["turnover"]["accion"] != "mantener").sum() for f in carteras
        )
        print(f"\nOperaciones totales en {n_rebalanceos} rebalanceos trimestrales: {n_operaciones_totales}")
        print(f"Media de operaciones por rebalanceo: {n_operaciones_totales / n_rebalanceos:.1f}")
        return

    ruta_salida_retornos = f"{RUTA_DATOS_PROCESADOS}/retorno_cartera_acciones{sufijo}.csv"
    retorno_cartera_acciones.to_csv(ruta_salida_retornos, header=["retorno_mensual"])
    print(f"\nRetornos guardados en: {ruta_salida_retornos}")

    ruta_salida_metricas = f"{RUTA_DATOS_PROCESADOS}/comparacion_cartera_acciones{sufijo}_vs_spy.csv"
    resultados.to_csv(ruta_salida_metricas)
    print(f"Métricas guardadas en: {ruta_salida_metricas}")

    if modo == "walk_forward":
        print("\n=== 7. Exportando histórico de turnover (Power BI + bootstrap de cartera vigente) ===")
        ruta_salida_turnover = f"{RUTA_DATOS_PROCESADOS}/{NOMBRE_FICHERO_HISTORICO_TURNOVER}"
        exportar_historico_turnover(carteras, ruta_salida_turnover)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modo",
        choices=["simple", "walk_forward", "trimestral"],
        default="walk_forward",
        help="Variante de backtest a ejecutar (por defecto: walk_forward).",
    )
    args = parser.parse_args()
    ejecutar(args.modo)
