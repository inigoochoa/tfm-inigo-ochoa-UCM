"""
Mapeo de todas las series de datos del proyecto.
"""

from datetime import date

FECHA_INICIO = "1990-01-01"
FECHA_FIN = date.today().isoformat()

SERIES_FRED = {
    "DGS10": "tipo_10y",
    "DGS2": "tipo_2y",
    "FEDFUNDS": "fed_funds",
    "BAA10Y": "spread_credito",
    "CPIAUCSL": "cpi",
    "INDPRO": "produccion_industrial",
    "UNRATE": "tasa_paro",
    "PPIACO": "ppi",
    "PSAVERT": "tasa_ahorro",
    "ICSA": "peticiones_paro",
    "CFNAI": "indice_ciclo",
}

TICKERS_YAHOO = {
    "^GSPC": "sp500",
    "^VIX": "vix",
}

DATASETS_KENNETH_FRENCH = {
    "F-F_Research_Data_5_Factors_2x3": "fama_french_5",
    "F-F_Momentum_Factor": "momentum",
}

URL_AQR_BAB = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx"

URL_AAII_SENTIMENT = "https://www.aaii.com/files/surveys/sentiment.xls"

URL_BASE_SHARADAR = "https://api.sharadar.com/v1.0/data"
TABLA_SHARADAR_SP500 = "sp500"
TABLA_SHARADAR_TICKERS = "tickers"
TABLA_SHARADAR_FUNDAMENTALS = "fundamentals"
TABLA_SHARADAR_STOCKS = "stocks"

RUTA_DATOS_CRUDOS = "data/raw"
RUTA_DATOS_PROCESADOS = "data/processed"
