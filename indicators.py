import pandas as pd
import numpy as np

def compute_sma(df: pd.DataFrame, window: int = 20, column: str = 'Close') -> pd.Series:
    """Computes the Simple Moving Average."""
    return df[column].rolling(window=window).mean()

def compute_ema(df: pd.DataFrame, window: int = 20, column: str = 'Close') -> pd.Series:
    """Computes the Exponential Moving Average."""
    return df[column].ewm(span=window, adjust=False).mean()

def compute_rsi(df: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.Series:
    """Computes the Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    # Calculate initial exponential moving averages for gains and losses
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close') -> tuple:
    """
    Computes the Moving Average Convergence Divergence (MACD).
    
    Returns:
    - Tuple of (macd_line, signal_line, macd_histogram)
    """
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist

def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2, column: str = 'Close') -> tuple:
    """
    Computes Bollinger Bands.
    
    Returns:
    - Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = df[column].rolling(window=window).mean()
    std_dev = df[column].rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band

def append_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends all key technical indicators to the DataFrame.
    """
    df_out = df.copy()
    
    # Moving Averages
    df_out['SMA_20'] = compute_sma(df_out, window=20)
    df_out['SMA_50'] = compute_sma(df_out, window=50)
    df_out['EMA_12'] = compute_ema(df_out, window=12)
    df_out['EMA_26'] = compute_ema(df_out, window=26)
    
    # Momentum
    df_out['RSI'] = compute_rsi(df_out, window=14)
    
    # Trend Convergence
    macd_line, signal_line, macd_hist = compute_macd(df_out)
    df_out['MACD'] = macd_line
    df_out['MACD_Signal'] = signal_line
    df_out['MACD_Hist'] = macd_hist
    
    # Volatility
    upper, mid, lower = compute_bollinger_bands(df_out)
    df_out['BB_Upper'] = upper
    df_out['BB_Middle'] = mid
    df_out['BB_Lower'] = lower
    
    return df_out

if __name__ == '__main__':
    # Test indicators calculation
    from data_loader import fetch_stock_data
    df = fetch_stock_data("AAPL", period="6mo")
    df_indicators = append_all_indicators(df)
    print(df_indicators[['Close', 'SMA_20', 'RSI', 'MACD', 'BB_Upper']].tail(10))
