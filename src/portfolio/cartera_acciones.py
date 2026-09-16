"""Cartera de acciones "core-satélite": SPY como núcleo pasivo + una tranche activa de acciones seleccionadas por momentum individual (retorno 12-1 meses), financiada comprando/vendiendo SPY."""

import pandas as pd
import yfinance as yf

from data_ingestion.universe import get_universe
from features.precios_pit import agregar_precios_mensual, combinar_fuentes_precios

NUCLEO_TICKER = "SPY"
N_ACCIONES_MOMENTUM = 15
MESES_FORMACION_MOMENTUM = 12
MESES_EXCLUSION_MOMENTUM = 1  # se salta el último mes (reversión a corto plazo)


def _descargar_spy_mensual(fecha_inicio: str, fecha_fin: str) -> pd.Series:
    # SPY es un ETF, no aparece en el panel combinado de Sharadar+Yahoo (acciones
    # individuales). Si se omite, calcular_retorno_cartera_acciones() lo trata
    # silenciosamente como "delistado" cada mes, corrompiendo el valor del núcleo.
    print(f"Descargando {NUCLEO_TICKER} vía Yahoo Finance (no está en el panel combinado)...")
    datos = yf.download(
        NUCLEO_TICKER, start=fecha_inicio, end=fecha_fin, progress=False, auto_adjust=True
    )
    precio_diario = datos["Close"]
    if isinstance(precio_diario, pd.DataFrame):
        precio_diario = precio_diario.iloc[:, 0]

    precio_mensual = precio_diario.resample("ME").last()
    precio_mensual.index.name = "fecha_mes"
    precio_mensual.name = NUCLEO_TICKER
    return precio_mensual


def cargar_precios_ticker(incluir_spy: bool = False) -> pd.DataFrame:
    # A diferencia de precios_pit.parquet, aquí NO se restringe al universo
    # point-in-time: el cálculo de momentum necesita el historial completo de cada
    # acción, incluidos los meses previos a su entrada en el índice. El filtro de
    # universo se aplica después, en seleccionar_top_momentum().
    # incluir_spy=True es obligatorio en cualquier llamada que alimente
    # calcular_retorno_cartera_acciones(): si se omite, SPY queda fuera de
    # precios.columns y se trata silenciosamente como "delistado" cada mes.
    precios_diarios = combinar_fuentes_precios()
    precios_mensuales = agregar_precios_mensual(precios_diarios)

    precios_ancho = precios_mensuales.pivot(
        index="fecha_mes", columns="ticker", values="precio_ajustado"
    )
    precios_ancho = precios_ancho.sort_index()

    if incluir_spy and NUCLEO_TICKER not in precios_ancho.columns:
        precio_spy = _descargar_spy_mensual(
            fecha_inicio=str(precios_ancho.index.min().date()),
            fecha_fin=str(precios_ancho.index.max().date() + pd.Timedelta(days=1)),
        )
        precios_ancho = precios_ancho.join(precio_spy, how="left")
        print(
            f"{NUCLEO_TICKER} añadido. Cobertura: "
            f"{precios_ancho[NUCLEO_TICKER].notna().sum()} de {len(precios_ancho)} meses"
        )

    return precios_ancho


def calcular_retorno_12_1(precios: pd.DataFrame, fecha: pd.Timestamp) -> pd.Series:
    # 12 meses antes de 'fecha' hasta 1 mes antes (se excluye el último mes,
    # estándar académico para evitar reversión a corto plazo).
    posicion_fecha = precios.index.get_loc(fecha)
    posicion_inicio = posicion_fecha - MESES_FORMACION_MOMENTUM
    posicion_fin = posicion_fecha - MESES_EXCLUSION_MOMENTUM

    if posicion_inicio < 0:
        raise ValueError(
            f"No hay suficiente historial de precios antes de {fecha.date()} "
            f"para calcular momentum 12-1 (se necesitan {MESES_FORMACION_MOMENTUM} meses previos)."
        )

    precio_inicio = precios.iloc[posicion_inicio]
    precio_fin = precios.iloc[posicion_fin]

    return (precio_fin / precio_inicio) - 1


def seleccionar_top_momentum(
    precios: pd.DataFrame,
    fecha: pd.Timestamp,
    universo: list[str] | None = None,
    n: int = N_ACCIONES_MOMENTUM,
) -> pd.Series:
    # universo=None usa todas las columnas de 'precios' sin restricción -- solo para
    # tests; en producción siempre hay que restringir para evitar survivorship bias.
    momentum = calcular_retorno_12_1(precios, fecha)

    if universo is not None:
        momentum = momentum[momentum.index.intersection(universo)]

    momentum = momentum.dropna()

    return momentum.sort_values(ascending=False).head(n)


def construir_cartera_objetivo(
    seleccion_momentum: pd.Series, peso_tranche_momentum: float
) -> pd.Series:
    n = len(seleccion_momentum)
    if n == 0:
        # Sin selección válida ese mes (datos insuficientes): todo a SPY.
        return pd.Series({NUCLEO_TICKER: 1.0})

    peso_por_accion = peso_tranche_momentum / n
    pesos = pd.Series(peso_por_accion, index=seleccion_momentum.index)
    pesos[NUCLEO_TICKER] = 1.0 - peso_tranche_momentum

    return pesos


def calcular_turnover(
    cartera_nueva: pd.Series, cartera_anterior: pd.Series | None
) -> pd.DataFrame:
    # cartera_anterior=None (primer rebalanceo de la serie) se interpreta como 100% SPY.
    if cartera_anterior is None:
        cartera_anterior = pd.Series({NUCLEO_TICKER: 1.0})

    todos_los_activos = cartera_nueva.index.union(cartera_anterior.index)

    peso_nuevo = cartera_nueva.reindex(todos_los_activos, fill_value=0.0)
    peso_anterior = cartera_anterior.reindex(todos_los_activos, fill_value=0.0)
    delta = peso_nuevo - peso_anterior

    def clasificar(fila):
        if fila["peso_anterior"] == 0 and fila["peso_nuevo"] > 0:
            return "compra (nueva entrada)"
        if fila["peso_anterior"] > 0 and fila["peso_nuevo"] == 0:
            return "venta (sale de cartera)"
        if fila["delta"] > 1e-9:
            return "compra (aumenta peso)"
        if fila["delta"] < -1e-9:
            return "venta (reduce peso)"
        return "mantener"

    turnover = pd.DataFrame({
        "peso_anterior": peso_anterior,
        "peso_nuevo": peso_nuevo,
        "delta": delta,
    })
    turnover["accion"] = turnover.apply(clasificar, axis=1)

    return turnover.sort_values("delta", ascending=False)


def construir_universos_por_fecha(fechas_rebalanceo: pd.Index) -> dict[pd.Timestamp, list[str]]:
    return {
        fecha: get_universe(fecha, verbose=False)["ticker"].tolist()
        for fecha in fechas_rebalanceo
    }


def construir_serie_carteras(
    precios: pd.DataFrame,
    pesos_momentum_anuales: pd.Series,
    universos_por_fecha: dict[pd.Timestamp, list[str]] | None = None,
    n: int = N_ACCIONES_MOMENTUM,
) -> dict[pd.Timestamp, dict]:
    # 'pesos_momentum_anuales' debe venir muestreado solo en las fechas de rebalanceo
    # deseadas (pese al nombre, funciona con cualquier cadencia: ver
    # ejecutar_cartera_acciones.py --modo trimestral).
    resultado = {}
    cartera_anterior = None

    for fecha, peso_momentum in pesos_momentum_anuales.items():
        universo = universos_por_fecha.get(fecha) if universos_por_fecha else None

        seleccion = seleccionar_top_momentum(precios, fecha, universo=universo, n=n)
        cartera = construir_cartera_objetivo(seleccion, peso_momentum)
        turnover = calcular_turnover(cartera, cartera_anterior)

        resultado[fecha] = {"cartera": cartera, "turnover": turnover}
        cartera_anterior = cartera

    return resultado


def calcular_coste_transaccion(turnover: pd.DataFrame, coste_bps: float) -> float:
    # /2: turnover['delta'].abs().sum() cuenta compra y venta del mismo importe
    # reasignado (cartera autofinanciada); se divide para no contarlo dos veces.
    turnover_lado = turnover["delta"].abs().sum() / 2
    return turnover_lado * (coste_bps / 10_000)


def calcular_retorno_cartera_acciones(
    carteras_por_fecha: dict[pd.Timestamp, dict],
    precios: pd.DataFrame,
    meses_por_fold: int = 12,
    coste_bps: float = 0.0,
) -> pd.Series:
    # Si una acción deja de tener precio antes del siguiente rebalanceo (delisting),
    # su valor acumulado se reinvierte en SPY para el resto del periodo, sin
    # redistribuir entre las demás acciones ni congelarlo.
    # coste_bps: valor_total_anterior queda fijo en el capital pre-costo (1.0)
    # mientras que valor_por_activo sí lleva el descuento, para que el coste se
    # manifieste como una caída real de retorno en el primer mes.
    fechas_rebalanceo = sorted(carteras_por_fecha.keys())
    retornos_mensuales = {}

    for fecha_inicio in fechas_rebalanceo:
        posicion_inicio = precios.index.get_loc(fecha_inicio)
        posicion_fin = min(posicion_inicio + meses_por_fold, len(precios.index) - 1)
        meses_periodo = precios.index[posicion_inicio + 1 : posicion_fin + 1]

        if len(meses_periodo) == 0:
            continue

        cartera_objetivo = carteras_por_fecha[fecha_inicio]["cartera"]
        turnover = carteras_por_fecha[fecha_inicio]["turnover"]

        coste_fraccion = calcular_coste_transaccion(turnover, coste_bps) if coste_bps > 0 else 0.0

        valor_total_anterior = cartera_objetivo.sum()  # = 1.0, pre-costo
        valor_por_activo = cartera_objetivo.copy() * (1 - coste_fraccion)  # post-costo

        for mes in meses_periodo:
            for activo, valor in list(valor_por_activo.items()):
                if valor == 0:
                    continue

                precio_actual = precios.at[mes, activo] if activo in precios.columns else None

                if precio_actual is None or pd.isna(precio_actual):
                    valor_por_activo[NUCLEO_TICKER] = valor_por_activo.get(NUCLEO_TICKER, 0.0) + valor
                    valor_por_activo[activo] = 0.0
                    continue

                mes_anterior_idx = precios.index.get_loc(mes) - 1
                precio_mes_anterior = precios.iloc[mes_anterior_idx][activo]

                if pd.isna(precio_mes_anterior):
                    continue

                retorno_activo = (precio_actual / precio_mes_anterior) - 1
                valor_por_activo[activo] = valor * (1 + retorno_activo)

            valor_total_nuevo = valor_por_activo.sum()
            retorno_cartera_mes = (valor_total_nuevo / valor_total_anterior) - 1

            retornos_mensuales[mes] = retorno_cartera_mes
            valor_total_anterior = valor_total_nuevo

    return pd.Series(retornos_mensuales).sort_index()


def exportar_historico_turnover(
    carteras_por_fecha: dict[pd.Timestamp, dict], ruta_salida: str
) -> pd.DataFrame:
    # Aplana a un DataFrame largo (una fila por ticker por rebalanceo). Lo consumen
    # Power BI y calcular_cartera_vigente.py (bootstrap de la cartera "anterior").
    piezas = []
    for fecha, contenido in sorted(carteras_por_fecha.items()):
        turnover = contenido["turnover"].copy()
        turnover.index.name = "ticker"
        turnover = turnover.reset_index()
        turnover.insert(0, "fecha_rebalanceo", fecha)
        piezas.append(turnover)

    historico = pd.concat(piezas, ignore_index=True)
    historico.to_csv(ruta_salida, index=False)

    print(
        f"Histórico de turnover guardado en {ruta_salida} — "
        f"{len(historico)} filas, {historico['fecha_rebalanceo'].nunique()} rebalanceos"
    )

    return historico

def reconstruir_carteras_por_fecha(ruta_turnover: str) -> dict[pd.Timestamp, dict]:
    # Inversa de exportar_historico_turnover(). La 'cartera' de cada fecha son los
    # tickers con peso_nuevo > 0 (mismo criterio que construir_serie_carteras()).
    historico = pd.read_csv(ruta_turnover, parse_dates=["fecha_rebalanceo"])

    carteras_por_fecha = {}
    for fecha, grupo in historico.groupby("fecha_rebalanceo"):
        turnover = grupo.set_index("ticker")[["peso_anterior", "peso_nuevo", "delta", "accion"]]
        cartera = turnover.loc[turnover["peso_nuevo"] > 0, "peso_nuevo"]
        carteras_por_fecha[fecha] = {"cartera": cartera, "turnover": turnover}

    return carteras_por_fecha