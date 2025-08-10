# refactored_fyers_swing/config.py

import os
from pathlib import Path

# === FYERS API ===
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "VE3CCLJZWA-100" )
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID", "QEGA69PVUL")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://www.google.com" )
FYERS_APP_ID_HASH = os.getenv("FYERS_APP_ID_HASH", "b209632623b60de416ea3bcbd2b780ef11ebdbb652b3f06f63ffdd34366faa18")

TOKEN_PATH = Path("outputs/fyers_access_token.txt")


#=====Good Stock Filter====
AVG_VOL = 3
AVG_PRICE = 100

# === DATA SETTINGS ===
SYMBOLS_FILE = Path("symbols.csv")
HISTORICAL_DATA_FILE = Path("outputs/all_symbols_history.parquet")
HISTORICAL_DATA_FILE_csv = Path("outputs/all_symbols_history.csv")
LATEST_DATA_FILE = Path("outputs/latest_full_data.parquet")
DAILY_DATA_FILE = Path("outputs/today_data.csv")


# === INDICATOR SETTINGS ===
EMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20

# === MODEL ===
version = input("Please enter model verison")
MODEL_FILE = Path(f"models/{version}/enhanced_model_pipeline.pkl")

# === LOGGING ===
LOG_FILE = Path("outputs/logs/system.log")

#====Feature Columns====
FEATURE_COLUMNS = [
    'ema20_ema50',	'ema50_ema200',	'ema20_price',	'ema50_price',	'ema200_price',	'rsi',	'atr_pct',
    'obv',	'vol_change_5d', 'ret_1d',	'ret_3d',	'ret_5d',	'ret_10d', 'vol_5d',	
    'vol_10d',	'rsi_3_slope',	'rsi_5_slope',	'ema20_above_ema50', 'ema20_50_cross_up', 'ema20_50_cross_down',
    'close_position_in_range',	'gap_pct',	'fib_pivot_distance_pct',	'fib_r1_distance_pct',	'fib_r2_distance_pct',
    'fib_s1_distance_pct',	'fib_s2_distance_pct', 'is_bullish',	'is_bearish',	'body_to_range',	
    'upper_shadow_to_range',	'lower_shadow_to_range',	'is_doji',	'is_hammer',	'is_shooting_star',
    'close_compared_to_previous', 'bb_position', 'support_pct', 'resistance_pct'
]


CONFIDENCE_THRESHOLD = 0.65


