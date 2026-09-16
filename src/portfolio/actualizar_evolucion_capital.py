"""Regenera evolucion_capital_100k.csv (SPY puro, cartera con 5 pb de coste, cartera sin costes) sobre la serie de retornos completa."""

import pandas as pd

from config.series import RUTA_DATOS_PROCESADOS
from portfolio.cartera_acciones import (
    cargar_precios_ticker,
    calcular_retorno_cartera_acciones,
    reconstruir_carteras_por_fecha,
)

RUTA_TURNOVER_COMPLETO = f"{RUTA_DATOS_PROCESADOS}/historico_turnover_cartera_completo.csv"
RUTA_SALIDA = f"{RUTA_DATOS_PROCESADOS}/evolucion_capital_100k.csv"
RUTA_SALIDA_CARTERA_VS_SPY = f"{RUTA_DATOS_PROCESADOS}/capital_cartera_vs_spy.csv"

CAPITAL_INICIAL = 100_000.0
COSTE_BPS = 5.0
TICKER_SPY = "SPY"


def main() -> None:
    print("=== 1. Reconstruyendo carteras+turnover por fecha ===")
    carteras_por_fecha = reconstruir_carteras_por_fecha(RUTA_TURNOVER_COMPLETO)
    print(f"Rebalanceos reconstruidos: {len(carteras_por_fecha)}")

    print("\n=== 2. Cargando precios (una sola vez, para SPY y para ambos escenarios) ===")
    precios = cargar_precios_ticker(incluir_spy=True)

    # Se recalcula aquí en vez de reutilizar un CSV en disco, para que ambos escenarios
    # partan de la misma descarga de precios y no queden inconsistentes entre sí.
    print("\n=== 3. Calculando retorno SIN costes ===")
    retorno_sin_costes = calcular_retorno_cartera_acciones(
        carteras_por_fecha, precios, meses_por_fold=12, coste_bps=0.0
    )
    print(f"Periodo: {retorno_sin_costes.index.min().date()} a {retorno_sin_costes.index.max().date()} "
          f"({len(retorno_sin_costes)} meses)")

    print(f"\n=== 4. Calculando retorno con {COSTE_BPS:.0f} pb de coste (misma precios, mismas carteras) ===")
    retorno_con_costes = calcular_retorno_cartera_acciones(
        carteras_por_fecha, precios, meses_por_fold=12, coste_bps=COSTE_BPS
    )
    print(f"Periodo: {retorno_con_costes.index.min().date()} a {retorno_con_costes.index.max().date()} "
          f"({len(retorno_con_costes)} meses)")

    print("\n=== 5. Calculando retorno mensual de SPY en el mismo periodo ===")
    fecha_inicio_rebalanceo = min(carteras_por_fecha.keys())
    retorno_spy = precios[TICKER_SPY].pct_change().loc[retorno_sin_costes.index]
    print(f"Meses con retorno de SPY: {retorno_spy.notna().sum()} de {len(retorno_spy)}")

    print("\n=== 6. Construyendo evolución de $100.000 en los tres escenarios ===")
    fechas_completas = [fecha_inicio_rebalanceo] + list(retorno_sin_costes.index)

    capital_spy = [CAPITAL_INICIAL]
    capital_con_costes = [CAPITAL_INICIAL]
    capital_sin_costes = [CAPITAL_INICIAL]

    for fecha in retorno_sin_costes.index:
        capital_spy.append(capital_spy[-1] * (1 + retorno_spy.loc[fecha]))
        capital_con_costes.append(capital_con_costes[-1] * (1 + retorno_con_costes.loc[fecha]))
        capital_sin_costes.append(capital_sin_costes[-1] * (1 + retorno_sin_costes.loc[fecha]))

    evolucion = pd.DataFrame({
        "fecha_mes": fechas_completas,
        "capital_spy": capital_spy,
        f"capital_cartera_{COSTE_BPS:.0f}pb": capital_con_costes,
        "capital_cartera_sin_costes": capital_sin_costes,
    }).set_index("fecha_mes")

    evolucion.to_csv(RUTA_SALIDA)
    print(f"\nGuardado en {RUTA_SALIDA} -- {len(evolucion)} filas "
          f"({evolucion.index.min().date()} a {evolucion.index.max().date()})")

    evolucion[[f"capital_cartera_{COSTE_BPS:.0f}pb", "capital_spy"]].rename(
        columns={f"capital_cartera_{COSTE_BPS:.0f}pb": "capital_cartera"}
    ).to_csv(RUTA_SALIDA_CARTERA_VS_SPY)
    print(f"Guardado en {RUTA_SALIDA_CARTERA_VS_SPY} "
          f"(capital_cartera = con {COSTE_BPS:.0f} pb de coste, misma serie que la Figura 4)")

    print("\n=== Resumen final ===")
    fila_final = evolucion.iloc[-1]
    print(f"SPY puro: ${fila_final['capital_spy']:,.0f}")
    print(f"Cartera con {COSTE_BPS:.0f} pb: ${fila_final[f'capital_cartera_{COSTE_BPS:.0f}pb']:,.0f}")
    print(f"Cartera sin costes: ${fila_final['capital_cartera_sin_costes']:,.0f}")


if __name__ == "__main__":
    main()