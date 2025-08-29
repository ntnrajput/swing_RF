import pandas as pd
import numpy as np
import talib

def add_macd_features(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Add comprehensive MACD features to the dataframe
    
    Parameters:
    - df: DataFrame with 'close' column
    - fast_period: Fast EMA period (default 12)
    - slow_period: Slow EMA period (default 26)  
    - signal_period: Signal line EMA period (default 9)
    
    Returns:
    - DataFrame with added MACD features
    """
    
    df = df.copy()
    
    # ============================================
    # SECTION 1: Basic MACD Components
    # ============================================
    
    # Calculate MACD line, Signal line, and Histogram using TA-Lib
    macd_line, macd_signal, macd_histogram = talib.MACD(
        df['close'].values, 
        fastperiod=fast_period, 
        slowperiod=slow_period, 
        signalperiod=signal_period
    )
    
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal  
    df['macd_histogram'] = macd_histogram
    
    # ============================================
    # SECTION 2: MACD Momentum Features
    # ============================================
    
    # MACD Line momentum (rate of change)
    df['macd_line_roc_3'] = df['macd_line'].pct_change(3) * 100
    df['macd_line_roc_5'] = df['macd_line'].pct_change(5) * 100
    df['macd_line_roc_10'] = df['macd_line'].pct_change(10) * 100
    
    # MACD Signal momentum
    df['macd_signal_roc_3'] = df['macd_signal'].pct_change(3) * 100
    df['macd_signal_roc_5'] = df['macd_signal'].pct_change(5) * 100
    
    # MACD Histogram momentum
    df['macd_hist_roc_3'] = df['macd_histogram'].pct_change(3) * 100
    df['macd_hist_roc_5'] = df['macd_histogram'].pct_change(5) * 100
    
    # ============================================
    # SECTION 3: MACD Cross and Signal Features
    # ============================================
    
    # MACD Line vs Signal Line crossover signals
    df['macd_above_signal'] = (df['macd_line'] > df['macd_signal']).astype(int)
    df['macd_cross_above'] = ((df['macd_line'] > df['macd_signal']) & 
                             (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_below'] = ((df['macd_line'] < df['macd_signal']) & 
                             (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    
    # MACD vs Zero Line signals
    df['macd_above_zero'] = (df['macd_line'] > 0).astype(int)
    df['macd_cross_above_zero'] = ((df['macd_line'] > 0) & 
                                  (df['macd_line'].shift(1) <= 0)).astype(int)
    df['macd_cross_below_zero'] = ((df['macd_line'] < 0) & 
                                  (df['macd_line'].shift(1) >= 0)).astype(int)
    
    # Histogram signals (positive/negative)
    df['macd_hist_positive'] = (df['macd_histogram'] > 0).astype(int)
    df['macd_hist_increasing'] = (df['macd_histogram'] > df['macd_histogram'].shift(1)).astype(int)
    
    # ============================================
    # SECTION 4: MACD Divergence Features
    # ============================================
    
    # Price vs MACD divergence detection (simplified)
    price_highs = df['close'].rolling(5).max()
    price_lows = df['close'].rolling(5).min()
    macd_highs = df['macd_line'].rolling(5).max()
    macd_lows = df['macd_line'].rolling(5).min()
    
    # Bullish divergence: price makes lower lows, MACD makes higher lows
    df['macd_bullish_divergence'] = ((df['close'] == price_lows) & 
                                    (df['macd_line'] > macd_lows.shift(5))).astype(int)
    
    # Bearish divergence: price makes higher highs, MACD makes lower highs  
    df['macd_bearish_divergence'] = ((df['close'] == price_highs) & 
                                    (df['macd_line'] < macd_highs.shift(5))).astype(int)
    
    # ============================================
    # SECTION 5: MACD Statistical Features
    # ============================================
    
    # MACD percentile ranks (momentum strength)
    df['macd_line_rank_20'] = df['macd_line'].rolling(20).rank(pct=True)
    df['macd_line_rank_50'] = df['macd_line'].rolling(50).rank(pct=True)
    df['macd_histogram_rank_20'] = df['macd_histogram'].rolling(20).rank(pct=True)
    
    # MACD volatility (standard deviation)
    df['macd_line_std_10'] = df['macd_line'].rolling(10).std()
    df['macd_line_std_20'] = df['macd_line'].rolling(20).std()
    df['macd_histogram_std_10'] = df['macd_histogram'].rolling(10).std()
    
    # MACD distance from signal (spread)
    df['macd_signal_spread'] = df['macd_line'] - df['macd_signal']
    df['macd_signal_spread_pct'] = ((df['macd_line'] - df['macd_signal']) / 
                                   abs(df['macd_signal']) * 100).replace([np.inf, -np.inf], 0)
    
    # ============================================
    # SECTION 6: MACD Trend Strength Features
    # ============================================
    
    # Consecutive periods above/below signal
    def count_consecutive(series):
        return series.groupby((series != series.shift(1)).cumsum()).cumsum()
    
    df['macd_consecutive_above_signal'] = count_consecutive(df['macd_above_signal'])
    df['macd_consecutive_positive_hist'] = count_consecutive(df['macd_hist_positive'])
    
    # MACD acceleration (second derivative)
    df['macd_line_acceleration'] = df['macd_line'].diff().diff()
    df['macd_histogram_acceleration'] = df['macd_histogram'].diff().diff()
    
    # ============================================
    # SECTION 7: MACD Normalization Features
    # ============================================
    
    # Normalized MACD (relative to recent price movements)
    df['macd_line_normalized'] = df['macd_line'] / df['close'].rolling(20).std()
    df['macd_histogram_normalized'] = df['macd_histogram'] / df['close'].rolling(20).std()
    
    # MACD relative to its own historical values
    df['macd_line_zscore_20'] = ((df['macd_line'] - df['macd_line'].rolling(20).mean()) / 
                                df['macd_line'].rolling(20).std())
    df['macd_line_zscore_50'] = ((df['macd_line'] - df['macd_line'].rolling(50).mean()) / 
                                df['macd_line'].rolling(50).std())
    
    # ============================================
    # SECTION 8: Multi-Timeframe MACD Features
    # ============================================
    
    # Alternative MACD periods for different timeframes
    # Fast MACD (8,17,9) - more sensitive
    macd_fast, signal_fast, hist_fast = talib.MACD(
        df['close'].values, fastperiod=8, slowperiod=17, signalperiod=9
    )
    df['macd_fast_line'] = macd_fast
    df['macd_fast_histogram'] = hist_fast
    df['macd_fast_above_signal'] = (macd_fast > signal_fast).astype(int)
    
    # Slow MACD (19,39,9) - more stable  
    macd_slow, signal_slow, hist_slow = talib.MACD(
        df['close'].values, fastperiod=19, slowperiod=39, signalperiod=9
    )
    df['macd_slow_line'] = macd_slow
    df['macd_slow_histogram'] = hist_slow
    df['macd_slow_above_signal'] = (macd_slow > signal_slow).astype(int)
    
    # Agreement between different MACD timeframes
    df['macd_timeframe_agreement'] = ((df['macd_above_signal'] == 1) & 
                                     (df['macd_fast_above_signal'] == 1) & 
                                     (df['macd_slow_above_signal'] == 1)).astype(int)
    
    print(f"✅ Added {len([col for col in df.columns if 'macd' in col.lower()])} MACD features")
    
    return df

# ============================================
# Alternative Implementation without TA-Lib
# ============================================

def add_macd_features_manual(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Manual MACD calculation without TA-Lib dependency
    """
    
    df = df.copy()
    
    # Calculate EMAs manually
    def calculate_ema(prices, span):
        return prices.ewm(span=span, adjust=False).mean()
    
    # Basic MACD components
    ema_fast = calculate_ema(df['close'], fast_period)
    ema_slow = calculate_ema(df['close'], slow_period)
    
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd_line'], signal_period)
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    
    # Add all the same features as above using the manually calculated MACD
    # (Same feature engineering code as in the TA-Lib version)
    
    # MACD Cross signals
    df['macd_above_signal'] = (df['macd_line'] > df['macd_signal']).astype(int)
    df['macd_cross_above'] = ((df['macd_line'] > df['macd_signal']) & 
                             (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    
    # MACD vs Zero
    df['macd_above_zero'] = (df['macd_line'] > 0).astype(int)
    
    # MACD momentum
    df['macd_line_roc_5'] = df['macd_line'].pct_change(5) * 100
    df['macd_histogram_roc_3'] = df['macd_histogram'].pct_change(3) * 100
    
    # MACD spread
    df['macd_signal_spread'] = df['macd_line'] - df['macd_signal']
    
    # MACD percentile ranks
    df['macd_line_rank_20'] = df['macd_line'].rolling(20).rank(pct=True)
    df['macd_histogram_rank_20'] = df['macd_histogram'].rolling(20).rank(pct=True)
    
    print(f"✅ Added {len([col for col in df.columns if 'macd' in col.lower()])} MACD features (manual calculation)")
    
    return df

# ============================================
# Usage Examples
# ============================================

def demo_usage():
    """
    Example of how to use the MACD feature functions
    """
    
    # Sample usage in your existing pipeline:
    
    # Option 1: With TA-Lib (recommended)
    # df_with_macd = add_macd_features(df)
    
    # Option 2: Without TA-Lib  
    # df_with_macd = add_macd_features_manual(df)
    
    # Option 3: Custom parameters
    # df_with_macd = add_macd_features(df, fast_period=8, slow_period=21, signal_period=5)
    
    # Add to your FEATURE_COLUMNS list in config.py:
    macd_features = [
        'macd_line', 'macd_signal', 'macd_histogram',
        'macd_above_signal', 'macd_cross_above', 'macd_above_zero',
        'macd_line_roc_5', 'macd_histogram_roc_3',
        'macd_signal_spread', 'macd_line_rank_20',
        'macd_histogram_rank_20', 'macd_timeframe_agreement'
        # Add more features as needed
    ]
    
    # Then update your config:
    # FEATURE_COLUMNS.extend(macd_features)
    
    print("MACD features ready to use!")

if __name__ == "__main__":
    demo_usage()