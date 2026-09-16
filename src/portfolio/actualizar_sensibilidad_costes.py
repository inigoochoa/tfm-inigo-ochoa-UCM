"""Regenera sensibilidad_costes_transaccion.csv (0/5/10/20/50 pb) sobre el periodo completo 2014-2026."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from portfolio.cartera_acciones import (
    cargar_precios_ticker,
    calcular_retorno_cartera_acciones,
    reconstruir_carteras_por_fecha,
)
from portfolio.evaluacion_cartera import calcular_metricas

RUTA_TURNOVER_COMPLETO = f"{RUTA_DATOS_PROCESADOS}/historico_turnover_cartera_completo.csv"
RUTA_SALIDA = f"{RUTA_DATOS_PROCESADOS}/sensibilidad_costes_transaccion.csv"

TICKER_SPY = "SPY"
NIVELES_COSTE_BPS = [0, 5, 10, 20, 50]


def main() -> None:
    print("=== 1. Reconstruyendo carteras+turnover por fecha ===")
    carteras_por_fecha = reconstruir_carteras_por_fecha(RUTA_TURNOVER_COMPLETO)
    print(f"Rebalanceos reconstruidos: {len(carteras_por_fecha)}")

    print("\n=== 2. Cargando precios (incluye SPY) ===")
    precios = cargar_precios_ticker(incluir_spy=True)

    retorno_spy = None
    filas_resultado = []

    for coste_bps in NIVELES_COSTE_BPS:
        print(f"\n=== 3. Calculando retorno con {coste_bps} pb de coste ===")
        retorno_cartera = calcular_retorno_cartera_acciones(
            carteras_por_fecha, precios, meses_por_fold=12, coste_bps=coste_bps
        )

        if retorno_spy is None:
            retorno_spy = precios[TICKER_SPY].pct_change().loc[retorno_cartera.index]

        metricas = calcular_metricas(retorno_cartera, f"Cartera ({coste_bps} pb/operación)")
        filas_resultado.append(metricas)

    filas_resultado.append(calcular_metricas(retorno_spy, "SPY puro (referencia, sin fricción)"))

    resultados = pd.DataFrame(filas_resultado).set_index("cartera")
    print("\n=== Sensibilidad a costes de transacción (2014-2026) ===\n")
    print(resultados.round(4).to_string())

    resultados.to_csv(RUTA_SALIDA)
    print(f"\nGuardado en {RUTA_SALIDA}")


if __name__ == "__main__":
    main()