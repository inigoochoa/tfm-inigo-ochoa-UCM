"""Regenera comparacion_cartera_acciones_walk_forward_vs_spy.csv (sin coste de transacción) sobre el periodo completo 2014-2026."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from portfolio.cartera_acciones import (
    cargar_precios_ticker,
    calcular_retorno_cartera_acciones,
    reconstruir_carteras_por_fecha,
)
from portfolio.evaluacion_cartera import calcular_metricas

RUTA_TURNOVER_COMPLETO = f"{RUTA_DATOS_PROCESADOS}/historico_turnover_cartera_completo.csv"
RUTA_SALIDA = f"{RUTA_DATOS_PROCESADOS}/comparacion_cartera_acciones_walk_forward_vs_spy.csv"

TICKER_SPY = "SPY"


def main() -> None:
    print("=== 1. Reconstruyendo carteras+turnover por fecha ===")
    carteras_por_fecha = reconstruir_carteras_por_fecha(RUTA_TURNOVER_COMPLETO)
    print(f"Rebalanceos reconstruidos: {len(carteras_por_fecha)}")

    print("\n=== 2. Cargando precios (incluye SPY) ===")
    precios = cargar_precios_ticker(incluir_spy=True)

    print("\n=== 3. Calculando retorno de la cartera (sin coste, para la comparación de métricas) ===")
    retorno_cartera = calcular_retorno_cartera_acciones(
        carteras_por_fecha, precios, meses_por_fold=12, coste_bps=0.0
    )
    print(f"Periodo: {retorno_cartera.index.min().date()} a {retorno_cartera.index.max().date()} "
          f"({len(retorno_cartera)} meses)")

    print("\n=== 4. Calculando retorno de SPY en el mismo periodo ===")
    retorno_spy = precios[TICKER_SPY].pct_change().loc[retorno_cartera.index]

    print("\n=== 5. Calculando métricas comparadas ===")
    periodo_label = f"{retorno_cartera.index.min().year}-{retorno_cartera.index.max().year}"
    resultados = pd.DataFrame([
        calcular_metricas(retorno_cartera, f"Cartera core-satélite ({periodo_label})"),
        calcular_metricas(retorno_spy, "SPY puro, mismo periodo"),
    ]).set_index("cartera")

    print(resultados.round(4).to_string())

    resultados.to_csv(RUTA_SALIDA)
    print(f"\nGuardado en {RUTA_SALIDA}")


if __name__ == "__main__":
    main()