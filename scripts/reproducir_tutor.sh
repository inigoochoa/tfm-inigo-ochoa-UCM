#!/usr/bin/env bash
# Reproduce el proyecto de principio a fin SIN acceso a Sharadar: usa los
# modelos ya entrenados (models/) y las salidas ya calculadas (data/processed/)
# que vienen incluidos en el repo. No requiere SHARADAR_API_KEY en ningún
# momento. Pensado para ejecutarse desde la raíz del repo tras un git clone.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/4: comprobando ficheros necesarios ==="
FICHEROS_REQUERIDOS=(
    "models/rf_regimen_capa1.joblib"
    "models/rf_corto_final.joblib"
    "models/rf_medio_final.joblib"
    "models/rf_largo_final.joblib"
    "data/processed/panel_mensual_con_regimen.csv"
    "data/processed/evaluacion_capa2_vs_baselines.csv"
    "data/processed/evolucion_capital_100k.csv"
    "data/processed/prediccion_vigente.json"
    "data/processed/turnover_vigente.json"
)
FALTAN=0
for f in "${FICHEROS_REQUERIDOS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  FALTA: $f"
        FALTAN=1
    fi
done
if [ "$FALTAN" -eq 1 ]; then
    echo ""
    echo "Faltan ficheros del repo (ver arriba). No continúo: revisa que el" >&2
    echo "clon se ha hecho completo y que estos ficheros están versionados." >&2
    exit 1
fi
echo "OK: todos los ficheros necesarios están presentes."

echo ""
echo "=== 2/4: instalando dependencias ==="
pip install --quiet -r requirements.txt
pip install --quiet -e .
echo "OK: dependencias instaladas."

echo ""
echo "=== 3/4: comprobando que los modelos cargan y levantando la API ==="
python3 -c "
import joblib
for f in [
    'models/rf_regimen_capa1.joblib',
    'models/rf_corto_final.joblib',
    'models/rf_medio_final.joblib',
    'models/rf_largo_final.joblib',
]:
    modelo = joblib.load(f)
    print(f'  {f}: {type(modelo).__name__} OK')
"

PUERTO=8123
uvicorn api.main:app --port "$PUERTO" > /tmp/tfm_api_tutor.log 2>&1 &
PID_API=$!
trap 'kill "$PID_API" 2>/dev/null || true' EXIT

echo "Esperando a que la API arranque (PID $PID_API)..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null "http://127.0.0.1:$PUERTO/health"; then
        break
    fi
    sleep 1
done

echo "  GET /health:"
curl -sf "http://127.0.0.1:$PUERTO/health" && echo ""
echo "  GET /prediccion-factores:"
curl -sf "http://127.0.0.1:$PUERTO/prediccion-factores" && echo ""
echo "  GET /cartera-acciones (primeros 200 caracteres):"
curl -sf "http://127.0.0.1:$PUERTO/cartera-acciones" | head -c 200 && echo ""

kill "$PID_API" 2>/dev/null || true
trap - EXIT
echo "OK: API respondió sin necesidad de POST /recalcular-* ni de Sharadar."

echo ""
echo "=== 4/4: ejecutando el notebook de recorrido ==="
NOTEBOOK=$(find notebooks -maxdepth 1 -iname "*Notebook_Recorrido.ipynb" | head -1)
if [ -z "$NOTEBOOK" ]; then
    echo "No se encontró el notebook de recorrido en notebooks/." >&2
    exit 1
fi
jupyter nbconvert --to notebook --execute --inplace "$NOTEBOOK"
echo "OK: notebook ejecutado sin errores -> $NOTEBOOK"
echo "Figuras regeneradas en reports/figuras/"

echo ""
echo "=== Reproducción completa. Sin acceso a Sharadar en ningún paso. ==="
