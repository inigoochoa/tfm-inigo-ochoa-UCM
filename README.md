# Detección de regímenes de mercado con machine learning aplicada al asset allocation táctico

TFM — Máster en Data Science, Big Data & Business Analytics - Universidad Complutense de Madrid
Íñigo Ochoa Gil

## Objetivo

Detectar el régimen macro-financiero vigente mediante clustering no supervisado, usarlo junto
con señales cross-sectional como input para predecir qué factores de
Fama-French (+ Momentum + BAB) serán más rentables a distintos horizontes (corto,
medio, largo plazo), y traducir esas predicciones en una cartera de renta variable
_long-only_ sobre el S&P500.

## Arquitectura del pipeline

```
Datos (Yahoo Finance, FRED, Kenneth French, AQR, AAII, Sharadar)
        │
        ▼
Capa 1 — Detección de régimen
  clustering (k-means, k=4) sobre 19 variables macro/mercado → etiquetas de
                                régimen históricas (Expansión, Expansión
                                tardía/Endurecimiento, Crisis aguda, Recuperación)
  Random Forest + SHAP       → clasifica el régimen de un mes nuevo y lo explica

        │
        ▼
Capa 2 — Predicción de factores (Random Forest, uno por horizonte)
  Input:  variables macro + señales cross-sectional
          (relative strength, valuation composite, dispersion, value spread)

  Output: retorno esperado de los 7 factores en 3 horizontes (~1, ~3, ~12 meses)
        │
        ▼
Construcción de cartera (core-satellite: SPY pasivo + satélite top-15 por Momentum)
        │
        ▼
Backtesting (walk-forward / expanding window)
  Sharpe ratio · Máximo drawdown · Calmar ratio
        │
        ├──► Batch mensual
        └──► API (FastAPI)
              GET /health               → estado del servicio
              GET /prediccion-factores → sirve la predicción/tilt vigente (con caché)
              POST /recalcular-factores → fuerza recálculo, protegido con X-API-Key
              GET /cartera-acciones    → sirve el turnover de acciones vigente (con caché)
              POST /recalcular-cartera → fuerza recálculo, protegido con X-API-Key
```

## Factores objetivo

7 factores:
Fama-French 5 (Mkt-RF, SMB, HML, RMW, CMA) + Momentum (Kenneth French Data
Library) + BAB, Betting Against Beta (AQR Data Library)

## Universo de acciones y alcance

S&P500, sobre sus constituyentes _point-in-time_ (evitando sesgo de supervivencia). La
selección de universo está aislada en un único punto del código para poder
ampliarla en el futuro sin refactorizar el resto del pipeline.

## Rango temporal

El rango efectivo del análisis es 1990–2026 para las variables macro y de
mercado de la Capa 1, y 1998–2026 para las señales cross-sectional y el panel
de entrenamiento de la Capa 2, acotado por la cobertura real de la tabla de
constituyentes históricos del S&P500 de Sharadar. Las series de frecuencia
semanal (solicitudes de subsidio de desempleo y encuesta AAII) se agregan a
mensual antes de integrarse en el panel.

## Estructura del repositorio

```
README.md
pyproject.toml
requirements.txt
.env / .env.example

config/             Mapeo de series de datos y rutas de datos
data/               raw / processed / external (no versionados en git salvo
                    5 salidas agregadas necesarias para la Opción B, ver abajo)
models/             Modelos ya entrenados (régimen Capa 1 + 3 RF de Capa 2 por
                    horizonte), para reproducir sin necesidad de Sharadar
src/
  data_ingestion/   Descarga de mercado, macro, factores y fundamentales
  features/         Construcción de features de régimen y cross-sectional
  preprocessing/     Alineación de series
  regime/           Clustering (k-means) y clasificador RF+SHAP
  factor_prediction/  Target de retornos futuros y modelos por horizonte
  backtesting/      Walk-forward / expanding window (Capa 2)
  portfolio/        Construcción de cartera, pesos, evolución de capital,
                    sensibilidad a costes
api/                API FastAPI (predicción/tilt y cartera de acciones)
scripts/            reproducir_tutor.sh (Opción B) y utilidades de
                    entrenamiento/exportación de modelos
viz_exports/        PNGs de apoyo
notebooks/          Notebook de recorrido que reproduce las 5 figuras
                    principales del TFM a partir de los artefactos ya
                    calculados por src/ (no reimplementa el pipeline)
reports/
  figuras/          Gráficos finales de la memoria (regímenes, SHAP,
                    R²/acierto por factor, evolución de capital,
                    sensibilidad a costes) + gráficos del EDA
```

## Fuentes de datos por fase

| Fase                        | Fundamentales                                                          | Precios       | Macro / Factores                |
| --------------------------- | ---------------------------------------------------------------------- | ------------- | ------------------------------- |
| Entrenamiento / backtesting | Sharadar (point-in-time)                                               | Yahoo Finance | FRED, Kenneth French, AQR, AAII |
| Producción (tras entrega)   | SEC EDGAR (obligatorio, sin registro) + Finnhub (opcional, validación) | Yahoo Finance | FRED, Kenneth French, AQR, AAII |

Sharadar se usa solo para construir el dataset histórico de entrenamiento —
el modelo entrenado no depende de la suscripción una vez completada esa fase.

## Cómo ejecutar

```bash
conda create -n tfm-regimen python=3.11 -y
conda activate tfm-regimen
pip install -e .
pip install -r requirements.txt

cp .env.example .env                # Rellenar FRED_API_KEY (obligatoria para
                                    # la Opción A), TFM_API_KEY (protege los
                                    # POST de la API) y, si se usa, FINNHUB_API_KEY
```

### Opción A — Réplica completa (requiere suscripción a Sharadar)

Regenera los datos y reentrena los modelos desde cero. Además de lo anterior,
rellena `SHARADAR_API_KEY` en el `.env` — sin ella, este bloque falla.

```bash
# Tracking de experimentos (opcional, abre panel local)
mlflow ui

# 1. Ingesta de datos crudos (-> data/raw/)
python -m src.data_ingestion.fred_loader
python -m src.data_ingestion.loaders                # AQR BAB + Kenneth French + AAII

python -c "from data_ingestion.sharadar_loader import guardar_universo_sharadar; guardar_universo_sharadar()"
python -c "from data_ingestion.sharadar_loader import guardar_fundamentals_arq; guardar_fundamentals_arq()"
python -c "from data_ingestion.sharadar_loader import guardar_precios_delistados; guardar_precios_delistados()"

python -c "from data_ingestion.yahoo_loader import descargar_todas_series_yahoo; descargar_todas_series_yahoo()"
python -c "from data_ingestion.yahoo_loader import guardar_precios_individuales_yahoo; guardar_precios_individuales_yahoo()"

# 2. Preprocesamiento y Capa 1 (régimen) (-> data/processed/)
python -m src.preprocessing.alineacion
python -m src.features.macro
python -m src.features.cross_sectional
python -m src.regime.kmeans_regimen
python -m src.regime.regimen_sin_leakage

# 3. Features de Capa 2
python -m src.features.fundamentales_pit
python -m src.features.precios_pit
python -m src.features.pipeline_senales_capa2
python -m src.features.panel_maestro_capa2

# 4. Target y entrenamiento de Capa 2 (requiere mlflow ui activo o tracking configurado)
python -m src.factor_prediction.target
python -m src.factor_prediction.entrenamiento       # registra rf_corto_final / rf_medio_final / rf_largo_final en MLflow

# 5. Backtesting y cartera
python -m src.backtesting.walk_forward_capa2
python -m src.portfolio.pesos_cartera
python -m src.portfolio.ejecutar_cartera_acciones --modo walk_forward
python -m src.portfolio.extender_historico_reciente

# Levantar la API
uvicorn api.main:app --reload
# Documentación interactiva en http://127.0.0.1:8000/docs

# Precalcula la caché que sirven los GET (requiere TFM_API_KEY en .env)
curl -X POST http://127.0.0.1:8000/recalcular-factores -H "X-API-Key: $TFM_API_KEY"
curl -X POST http://127.0.0.1:8000/recalcular-cartera  -H "X-API-Key: $TFM_API_KEY"
```

Sharadar requiere suscripción de pago y el paso de `fundamentals` descarga
~620MB, por lo que ese tramo es el más costoso de replicar de cero. El resto
de fuentes (FRED, Yahoo Finance, Kenneth French, AQR, AAII, y SEC
EDGAR/Finnhub en producción) son de acceso gratuito.

### Opción B — Reproducir con los modelos ya entrenados (sin Sharadar)

Para verificar los resultados **sin necesitar ninguna suscripción de Sharadar**,
usando los modelos ya entrenados (`models/`) y las salidas ya calculadas
(`data/processed/`) incluidos en el repo. No hace falta `SHARADAR_API_KEY` ni
`FRED_API_KEY`.

```bash
bash scripts/reproducir_tutor.sh
```

El script comprueba que están los ficheros necesarios, instala dependencias,
carga los 4 modelos ya entrenados, levanta la API sirviendo desde caché,
valida los endpoints `GET`, y ejecuta el notebook de recorrido regenerando
las 5 figuras, sin acceder a Sharadar en ningún paso.

## Resultados principales

Solo Momentum a 12 meses muestra precisión direccional estadísticamente
significativa (R² = 0,174, precisión direccional = 70,8 %, p = 0,0011
binomial). El resto de combinaciones factor/horizonte no supera de forma
significativa a los benchmarks (predicción constante en cero y
predicción constante igual a la media histórica de entrenamiento). El
régimen de mercado, aunque válido como clasificador independiente en la
Capa 1, no aporta señal incremental en la Capa 2.

Cartera final (histórico hasta agosto 2026, con costes de 5 pb) vs. SPY:

| Métrica                   | Cartera  | SPY (benchmark) |
| ------------------------- | -------- | --------------- |
| Retorno anualizado (CAGR) | ~15,2 %  | ~14,2 %         |
| Sharpe                    | ~1,003   | ~0,991          |
| Calmar                    | ~0,661   | ~0,593          |
| Máximo drawdown           | ~-23,0 % | ~-23,9 %        |

La ventaja se mantiene en los cinco escenarios de coste evaluados (0, 5, 10,
20 y 50 pb por operación): incluso en el escenario más pesimista (50 pb), el
Sharpe de la cartera (0,998) sigue por encima del de SPY puro (0,991).

## Bibliografía principal

- "The Promises and Pitfalls of Factor Timing." _The Journal of Portfolio
  Management_, 44(4) (2018).
- "Factor Timing with Cross-Sectional and Time-Series Predictors." _The
  Journal of Portfolio Management_ (2017).
- "Machine Learning and Data Science Blueprints for Finance." O'Reilly Media
  (2020).
- "Advances in Financial Machine Learning." Wiley (2018).
- "Empirical Asset Pricing via Machine Learning." _The Review of Financial
  Studies_ (2020).
- "A Unified Approach to Interpreting Model Predictions." _Advances in Neural
  Information Processing Systems_ (2017).
- "Returns to Buying Winners and Selling Losers: Implications for Stock
  Market Efficiency." _The Journal of Finance_ (1993).
