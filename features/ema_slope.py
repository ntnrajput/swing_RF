import pandas as pd
import numpy as np
from scipy import stats

def add_ema_slope_features(df, ema_periods=[9, 20, 50, 100, 200]):
    """
    Add comprehensive EMA slope features to detect trend direction and strength
    
    Parameters:
    - df: DataFrame with 'close' column and existing EMA columns
    - ema_periods: List of EMA periods to analyze slopes for
    
    Returns:
    - DataFrame with added EMA slope features
    """
    
    df = df.copy()
    
    # Ensure we have the EMA columns (calculate if missing)
    for period in ema_periods:
        ema_col = f'ema{period}'
        if ema_col not in df.columns:
            df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()
            print(f"📊 Calculated missing {ema_col}")
    
    # ============================================
    # SECTION 1: Basic EMA Slopes (Percentage Change)
    # ============================================
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Short-term slopes (1, 2, 3 periods)
        df[f'{ema_col}_slope_1'] = df[ema_col].pct_change(1) * 100
        df[f'{ema_col}_slope_2'] = df[ema_col].pct_change(2) * 100  
        df[f'{ema_col}_slope_3'] = df[ema_col].pct_change(3) * 100
        
        # Medium-term slopes (5, 10 periods)
        df[f'{ema_col}_slope_5'] = df[ema_col].pct_change(5) * 100
        df[f'{ema_col}_slope_10'] = df[ema_col].pct_change(10) * 100
        
        # Long-term slopes (20 periods)
        if period <= 50:  # Avoid too long lookbacks for shorter EMAs
            df[f'{ema_col}_slope_20'] = df[ema_col].pct_change(20) * 100
    
    # ============================================
    # SECTION 2: Linear Regression Slope (True Slope)
    # ============================================
    
    def calculate_linear_slope(series, window):
        """Calculate linear regression slope over a rolling window"""
        def slope_func(y):
            if len(y) < 2 or y.isna().all():
                return np.nan
            x = np.arange(len(y))
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            try:
                slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
                return slope
            except:
                return np.nan
        
        return series.rolling(window=window).apply(slope_func, raw=False)
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Linear regression slopes over different windows
        df[f'{ema_col}_linslope_5'] = calculate_linear_slope(df[ema_col], 5)
        df[f'{ema_col}_linslope_10'] = calculate_linear_slope(df[ema_col], 10)
        df[f'{ema_col}_linslope_20'] = calculate_linear_slope(df[ema_col], 20)
    
    # ============================================
    # SECTION 3: EMA Direction Signals (Binary)
    # ============================================
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Direction indicators (1 = up, 0 = down)
        df[f'{ema_col}_rising_1'] = (df[ema_col] > df[ema_col].shift(1)).astype(int)
        df[f'{ema_col}_rising_3'] = (df[ema_col] > df[ema_col].shift(3)).astype(int)
        df[f'{ema_col}_rising_5'] = (df[ema_col] > df[ema_col].shift(5)).astype(int)
        df[f'{ema_col}_rising_10'] = (df[ema_col] > df[ema_col].shift(10)).astype(int)
        
        # Strong direction (rising for multiple consecutive periods)
        df[f'{ema_col}_strong_rising'] = ((df[ema_col] > df[ema_col].shift(1)) &
                                         (df[ema_col].shift(1) > df[ema_col].shift(2)) &
                                         (df[ema_col].shift(2) > df[ema_col].shift(3))).astype(int)
        
        df[f'{ema_col}_strong_falling'] = ((df[ema_col] < df[ema_col].shift(1)) &
                                          (df[ema_col].shift(1) < df[ema_col].shift(2)) &
                                          (df[ema_col].shift(2) < df[ema_col].shift(3))).astype(int)
    
    # ============================================
    # SECTION 4: EMA Slope Acceleration
    # ============================================
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # First derivative (slope)
        slope_1 = df[f'{ema_col}_slope_1']
        slope_3 = df[f'{ema_col}_slope_3']
        slope_5 = df[f'{ema_col}_slope_5']
        
        # Second derivative (acceleration) - slope of slope
        df[f'{ema_col}_acceleration_3'] = slope_1.rolling(3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 3, raw=False)
        df[f'{ema_col}_acceleration_5'] = slope_1.rolling(5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 5, raw=False)
        
        # Acceleration signals
        df[f'{ema_col}_accelerating_up'] = (df[f'{ema_col}_acceleration_3'] > 0).astype(int)
        df[f'{ema_col}_accelerating_down'] = (df[f'{ema_col}_acceleration_3'] < 0).astype(int)
    
    # ============================================
    # SECTION 5: EMA Slope Strength & Momentum
    # ============================================
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Slope magnitude (absolute value)
        df[f'{ema_col}_slope_magnitude_5'] = abs(df[f'{ema_col}_slope_5'])
        df[f'{ema_col}_slope_magnitude_10'] = abs(df[f'{ema_col}_slope_10'])
        
        # Slope consistency (how many of last N periods had same direction)
        def count_consistent_direction(series, window):
            return series.rolling(window).apply(
                lambda x: (np.sign(x) == np.sign(x.iloc[-1])).sum(), raw=False
            )
        
        df[f'{ema_col}_slope_consistency_5'] = count_consistent_direction(df[f'{ema_col}_slope_1'], 5)
        df[f'{ema_col}_slope_consistency_10'] = count_consistent_direction(df[f'{ema_col}_slope_1'], 10)
        
        # Slope percentile rank (relative strength)
        df[f'{ema_col}_slope_rank_20'] = df[f'{ema_col}_slope_5'].rolling(20).rank(pct=True)
        df[f'{ema_col}_slope_rank_50'] = df[f'{ema_col}_slope_5'].rolling(50).rank(pct=True)
    
    # ============================================
    # SECTION 6: Multi-EMA Slope Alignment
    # ============================================
    
    # All EMAs rising together (strong bullish alignment)
    rising_conditions = []
    for period in ema_periods:
        rising_conditions.append(df[f'ema{period}_rising_3'] == 1)
    
    df['all_emas_rising'] = np.logical_and.reduce(rising_conditions).astype(int)
    df['all_emas_falling'] = np.logical_and.reduce([~cond for cond in rising_conditions]).astype(int)
    
    # Majority EMA direction
    rising_sum = sum(rising_conditions)
    df['ema_majority_rising'] = (rising_sum >= len(ema_periods) // 2 + 1).astype(int)
    df['ema_rising_count'] = rising_sum
    
    # Short vs Long EMA slopes (momentum vs trend)
    if 20 in ema_periods and 50 in ema_periods:
        df['short_ema_stronger'] = (df['ema20_slope_5'] > df['ema50_slope_5']).astype(int)
    
    if 50 in ema_periods and 200 in ema_periods:
        df['medium_ema_stronger'] = (df['ema50_slope_5'] > df['ema200_slope_5']).astype(int)
    
    # ============================================
    # SECTION 7: EMA Slope Divergence
    # ============================================
    
    # Price vs EMA slope divergence
    price_slope_5 = df['close'].pct_change(5) * 100
    
    for period in [20, 50]:
        if period in ema_periods:
            ema_col = f'ema{period}'
            
            # Bullish divergence: price falling but EMA slope rising
            df[f'{ema_col}_bullish_divergence'] = ((price_slope_5 < 0) & 
                                                  (df[f'{ema_col}_slope_5'] > 0)).astype(int)
            
            # Bearish divergence: price rising but EMA slope falling  
            df[f'{ema_col}_bearish_divergence'] = ((price_slope_5 > 0) & 
                                                  (df[f'{ema_col}_slope_5'] < 0)).astype(int)
    
    # ============================================
    # SECTION 8: EMA Slope Normalized Features
    # ============================================
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Normalize slope by recent volatility
        price_volatility = df['close'].rolling(20).std()
        df[f'{ema_col}_slope_normalized'] = df[f'{ema_col}_slope_5'] / price_volatility
        
        # Z-score of slope (relative to recent slopes)
        slope_mean = df[f'{ema_col}_slope_5'].rolling(20).mean()
        slope_std = df[f'{ema_col}_slope_5'].rolling(20).std()
        df[f'{ema_col}_slope_zscore'] = (df[f'{ema_col}_slope_5'] - slope_mean) / slope_std
        
        # Slope momentum (slope getting stronger/weaker)
        df[f'{ema_col}_slope_momentum'] = df[f'{ema_col}_slope_5'].rolling(3).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
        )
    
    # ============================================
    # SECTION 9: EMA Slope Crossing Signals  
    # ============================================
    
    # When faster EMA slope crosses above slower EMA slope
    if len(ema_periods) >= 2:
        for i in range(len(ema_periods)-1):
            fast_period = ema_periods[i]
            slow_period = ema_periods[i+1]
            
            fast_slope = df[f'ema{fast_period}_slope_5']
            slow_slope = df[f'ema{slow_period}_slope_5']
            
            # Slope crossover signals
            df[f'ema{fast_period}_slope_above_ema{slow_period}'] = (fast_slope > slow_slope).astype(int)
            
            # Slope crossover events
            df[f'ema{fast_period}_slope_cross_above_ema{slow_period}'] = (
                (fast_slope > slow_slope) & 
                (fast_slope.shift(1) <= slow_slope.shift(1))
            ).astype(int)
    
    # ============================================
    # SECTION 10: EMA Angle Features (Degrees)
    # ============================================
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Convert slope to angle in degrees
        # Using arctangent of (price_change / time_periods)
        slope_radians = np.arctan(df[f'{ema_col}_slope_5'] / 100)  # Convert pct to decimal
        df[f'{ema_col}_angle_degrees'] = np.degrees(slope_radians)
        
        # Angle categories
        df[f'{ema_col}_steep_up'] = (df[f'{ema_col}_angle_degrees'] > 30).astype(int)      # > 30 degrees
        df[f'{ema_col}_moderate_up'] = ((df[f'{ema_col}_angle_degrees'] > 10) & 
                                       (df[f'{ema_col}_angle_degrees'] <= 30)).astype(int)  # 10-30 degrees
        df[f'{ema_col}_flat'] = (abs(df[f'{ema_col}_angle_degrees']) <= 10).astype(int)    # ±10 degrees
        df[f'{ema_col}_steep_down'] = (df[f'{ema_col}_angle_degrees'] < -30).astype(int)   # < -30 degrees
    
    # ============================================
    # SECTION 11: Summary Statistics
    # ============================================
    
    print(f"✅ Added {len([col for col in df.columns if 'slope' in col or 'rising' in col or 'acceleration' in col or 'angle' in col])} EMA slope features")
    
    # Print feature summary
    slope_features = [col for col in df.columns if any(x in col for x in ['slope', 'rising', 'acceleration', 'angle'])]
    print(f"📊 EMA Slope Feature Categories:")
    print(f"   - Basic slopes: {len([c for c in slope_features if 'slope_' in c and 'lin' not in c])}")
    print(f"   - Linear slopes: {len([c for c in slope_features if 'linslope' in c])}")
    print(f"   - Direction signals: {len([c for c in slope_features if 'rising' in c])}")
    print(f"   - Acceleration: {len([c for c in slope_features if 'acceleration' in c])}")
    print(f"   - Angle features: {len([c for c in slope_features if 'angle' in c])}")
    
    return df

# ============================================
# Simplified Version (Core Features Only)
# ============================================

def add_ema_slope_features_simple(df, ema_periods=[9, 20, 50, 200]):
    """
    Simplified version with only the most important EMA slope features
    """
    
    df = df.copy()
    
    # Ensure EMAs exist
    for period in ema_periods:
        ema_col = f'ema{period}'
        if ema_col not in df.columns:
            df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        
        # Core slope features
        df[f'{ema_col}_slope_3'] = df[ema_col].pct_change(3) * 100
        df[f'{ema_col}_slope_5'] = df[ema_col].pct_change(5) * 100
        df[f'{ema_col}_slope_10'] = df[ema_col].pct_change(10) * 100
        
        # Direction signals
        df[f'{ema_col}_rising'] = (df[ema_col] > df[ema_col].shift(3)).astype(int)
        
        # Strong directional moves
        df[f'{ema_col}_strong_up'] = ((df[f'{ema_col}_slope_5'] > 1.0) &  # > 1% slope
                                     (df[f'{ema_col}_rising'] == 1)).astype(int)
        
        df[f'{ema_col}_strong_down'] = ((df[f'{ema_col}_slope_5'] < -1.0) &  # < -1% slope
                                       (df[f'{ema_col}_rising'] == 0)).astype(int)
    
    # Multi-EMA alignment
    rising_conditions = [df[f'ema{period}_rising'] == 1 for period in ema_periods]
    df['all_emas_rising'] = np.logical_and.reduce(rising_conditions).astype(int)
    
    print(f"✅ Added {len([col for col in df.columns if 'slope' in col or 'rising' in col])} core EMA slope features")
    
    return df

# ============================================
# Usage Examples
# ============================================

def demo_usage():
    """
    Example of how to use the EMA slope feature functions
    """
    
    # Sample usage in your pipeline:
    
    # Option 1: Full feature set (recommended)
    # df_with_slopes = add_ema_slope_features(df, ema_periods=[9, 20, 50, 200])
    
    # Option 2: Simple version (faster)
    # df_with_slopes = add_ema_slope_features_simple(df)
    
    # Option 3: Custom EMA periods
    # df_with_slopes = add_ema_slope_features(df, ema_periods=[10, 21, 55, 100])
    
    # Key features to add to FEATURE_COLUMNS:
    key_slope_features = [
        # Core slopes
        'ema20_slope_5', 'ema50_slope_5', 'ema200_slope_5',
        # Direction signals  
        'ema20_rising', 'ema50_rising', 'ema200_rising',
        # Strong moves
        'ema20_strong_rising', 'ema50_strong_rising',
        # Multi-EMA alignment
        'all_emas_rising', 'ema_majority_rising',
        # Slope crossovers
        'ema20_slope_above_ema50', 'ema50_slope_above_ema200',
        # Angles
        'ema20_angle_degrees', 'ema50_angle_degrees'
    ]
    
    print("🚀 Top EMA Slope Features for Trading:")
    for feature in key_slope_features:
        print(f"   - {feature}")
    
    print("\n💡 These features will help identify:")
    print("   ✅ Trend direction and strength")
    print("   ✅ Momentum acceleration/deceleration") 
    print("   ✅ Multi-timeframe trend alignment")
    print("   ✅ Trend changes and reversals")

if __name__ == "__main__":
    demo_usage()