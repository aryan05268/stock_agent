import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, news_sentiment: float = 0.0) -> pd.DataFrame:
    """
    Analyzes technical indicators and sentiment to generate BUY, SELL, or HOLD signals.
    
    Parameters:
    - df: DataFrame containing price data and calculated indicators (SMA_20, RSI, MACD, etc.)
    - news_sentiment: Float representing the average news sentiment score (-1.0 to 1.0)
    
    Returns:
    - A DataFrame with columns containing technical signals, sentiment, and the agent's overall decision.
    """
    df_signals = df.copy()
    
    # 1. RSI Signal (-1 to 1)
    # RSI < 30 (Oversold) is Bullish. RSI > 70 (Overbought) is Bearish.
    df_signals['RSI_Signal'] = 0.0
    df_signals.loc[df_signals['RSI'] < 30, 'RSI_Signal'] = 1.0
    df_signals.loc[df_signals['RSI'] > 70, 'RSI_Signal'] = -1.0
    
    # 2. MACD Signal (-1 to 1)
    # We check for MACD crossovers: MACD crossing above MACD_Signal is bullish, below is bearish.
    df_signals['MACD_Signal_Val'] = 0.0
    # Crossover is detected when the sign of MACD_Hist changes
    for i in range(1, len(df_signals)):
        prev_hist = df_signals['MACD_Hist'].iloc[i - 1]
        curr_hist = df_signals['MACD_Hist'].iloc[i]
        
        if pd.isna(prev_hist) or pd.isna(curr_hist):
            continue
            
        if prev_hist < 0 and curr_hist > 0:
            df_signals.iloc[i, df_signals.columns.get_loc('MACD_Signal_Val')] = 1.0  # Bullish crossover
        elif prev_hist > 0 and curr_hist < 0:
            df_signals.iloc[i, df_signals.columns.get_loc('MACD_Signal_Val')] = -1.0 # Bearish crossover

    # 3. SMA Trend Signal (-1 to 1)
    # Price above SMA_20 is bullish, below is bearish
    df_signals['Trend_Signal'] = 0.0
    df_signals.loc[df_signals['Close'] > df_signals['SMA_20'], 'Trend_Signal'] = 0.5
    df_signals.loc[df_signals['Close'] < df_signals['SMA_20'], 'Trend_Signal'] = -0.5
    
    # Price above SMA_50 is additional bullish, below is additional bearish
    df_signals.loc[df_signals['Close'] > df_signals['SMA_50'], 'Trend_Signal'] += 0.5
    df_signals.loc[df_signals['Close'] < df_signals['SMA_50'], 'Trend_Signal'] -= 0.5
    
    # 4. Sentiment Signal (constant across recent window for simplicity)
    df_signals['Sentiment_Signal'] = 0.0
    if news_sentiment > 0.15:
        df_signals['Sentiment_Signal'] = 1.0
    elif news_sentiment < -0.15:
        df_signals['Sentiment_Signal'] = -1.0
        
    # 5. Combined Agent Score (-1.0 to 1.0)
    # Weights: RSI (25%), MACD (25%), Trend (20%), Sentiment (30%)
    w_rsi = 0.25
    w_macd = 0.25
    w_trend = 0.20
    w_sentiment = 0.30
    
    df_signals['Agent_Score'] = (
        (df_signals['RSI_Signal'] * w_rsi) +
        (df_signals['MACD_Signal_Val'] * w_macd) +
        (df_signals['Trend_Signal'] * w_trend) +
        (df_signals['Sentiment_Signal'] * w_sentiment)
    )
    
    # 6. Overall Action (BUY, SELL, HOLD)
    # Decision threshold: score >= 0.25 -> BUY, score <= -0.25 -> SELL, else HOLD
    df_signals['Signal_Action'] = 'HOLD'
    df_signals.loc[df_signals['Agent_Score'] >= 0.25, 'Signal_Action'] = 'BUY'
    df_signals.loc[df_signals['Agent_Score'] <= -0.25, 'Signal_Action'] = 'SELL'
    
    return df_signals

def get_latest_recommendation(df_signals: pd.DataFrame, ticker: str, sentiment_score: float) -> dict:
    """
    Generates a structured recommendation summary for the latest available data.
    """
    if df_signals.empty:
        return {}
        
    latest_row = df_signals.iloc[-1]
    
    # Construct explanation
    reasons = []
    
    # RSI check
    rsi = latest_row['RSI']
    if rsi < 30:
        reasons.append(f"RSI is oversold at {rsi:.1f} (bullish indicator).")
    elif rsi > 70:
        reasons.append(f"RSI is overbought at {rsi:.1f} (bearish indicator).")
    else:
        reasons.append(f"RSI momentum is neutral at {rsi:.1f}.")
        
    # MACD check
    macd_action = "neutral"
    macd_sig = latest_row['MACD_Signal_Val']
    if macd_sig == 1.0:
        reasons.append("A bullish MACD line crossover has occurred.")
    elif macd_sig == -1.0:
        reasons.append("A bearish MACD line crossover has occurred.")
        
    # Trend check
    trend = latest_row['Trend_Signal']
    close = latest_row['Close']
    sma20 = latest_row['SMA_20']
    if trend > 0:
        reasons.append(f"Stock price (${close:.2f}) is in an uptrend, trading above its 20-day SMA (${sma20:.2f}).")
    else:
        reasons.append(f"Stock price (${close:.2f}) is in a downtrend, trading below its 20-day SMA (${sma20:.2f}).")
        
    # Sentiment check
    if sentiment_score > 0.15:
        reasons.append(f"News sentiment is bullish with a compound score of {sentiment_score:.2f}.")
    elif sentiment_score < -0.15:
        reasons.append(f"News sentiment is bearish with a compound score of {sentiment_score:.2f}.")
    else:
        reasons.append(f"News sentiment is neutral at {sentiment_score:.2f}.")
        
    action = latest_row['Signal_Action']
    score = latest_row['Agent_Score']
    
    return {
        'ticker': ticker,
        'price': latest_row['Close'],
        'action': action,
        'score': score,
        'reasons': reasons,
        'timestamp': latest_row.name.strftime('%Y-%m-%d') if hasattr(latest_row.name, 'strftime') else str(latest_row.name)
    }

if __name__ == '__main__':
    # Test agent logic
    from data_loader import fetch_stock_data
    from indicators import append_all_indicators
    
    df = fetch_stock_data("AAPL", period="6mo")
    df_ind = append_all_indicators(df)
    df_sig = generate_signals(df_ind, news_sentiment=0.35)
    rec = get_latest_recommendation(df_sig, "AAPL", 0.35)
    
    print("\nLATEST RECOMMENDATION:")
    print(f"Action: {rec['action']} (Confidence Score: {rec['score']:.2f})")
    print("Reasons:")
    for r in rec['reasons']:
        print(f"- {r}")
