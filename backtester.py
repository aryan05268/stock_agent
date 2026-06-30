import pandas as pd
import numpy as np

def run_backtest(df_signals: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Simulates trading based on the agent's BUY/SELL/HOLD signals.
    
    Rules:
    - We start with a fixed amount of cash.
    - When signal is BUY and we have cash, we go 'All-In' (buy maximum possible shares).
    - When signal is SELL and we hold shares, we sell all shares.
    - Standard transaction cost is assumed to be 0% for simplicity, but can be configured.
    """
    cash = initial_capital
    position = 0.0  # Number of shares held
    portfolio_values = []
    trades = []
    
    # We drop any rows with NaN in indicators first so we start where signals are valid
    df_clean = df_signals.dropna(subset=['SMA_20', 'RSI', 'MACD']).copy()
    
    if df_clean.empty:
        return {'error': 'No data available to backtest after removing NaNs.'}
        
    buy_hold_shares = initial_capital / df_clean['Close'].iloc[0]
    
    for idx, row in df_clean.iterrows():
        close_price = row['Close']
        signal = row['Signal_Action']
        
        # 1. Process Signals
        if signal == 'BUY' and cash > 0:
            # Buy shares
            position = cash / close_price
            cash = 0.0
            trades.append({
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'type': 'BUY',
                'price': close_price,
                'portfolio_value': position * close_price
            })
        elif signal == 'SELL' and position > 0:
            # Sell shares
            cash = position * close_price
            position = 0.0
            trades.append({
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'type': 'SELL',
                'price': close_price,
                'portfolio_value': cash
            })
            
        # 2. Track daily portfolio value
        current_val = cash + (position * close_price)
        portfolio_values.append(current_val)
        
    df_clean['Portfolio_Value'] = portfolio_values
    df_clean['Buy_Hold_Value'] = buy_hold_shares * df_clean['Close']
    
    # 3. Performance Metrics
    final_value = portfolio_values[-1]
    strategy_return = ((final_value - initial_capital) / initial_capital) * 100
    
    bh_final_value = df_clean['Buy_Hold_Value'].iloc[-1]
    bh_return = ((bh_final_value - initial_capital) / initial_capital) * 100
    
    # Sharpe Ratio calculation (simplified annual Sharpe Ratio)
    # Daily returns
    df_clean['Daily_Return'] = df_clean['Portfolio_Value'].pct_change()
    mean_return = df_clean['Daily_Return'].mean()
    std_return = df_clean['Daily_Return'].std()
    
    if std_return > 0 and not pd.isna(std_return):
        # Annualized Sharpe Ratio = (Mean Daily Return / Std Daily Return) * sqrt(252 trading days)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        
    # Max Drawdown
    df_clean['Peak'] = df_clean['Portfolio_Value'].cummax()
    df_clean['Drawdown'] = (df_clean['Portfolio_Value'] - df_clean['Peak']) / df_clean['Peak']
    max_drawdown = df_clean['Drawdown'].min() * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'strategy_return_pct': strategy_return,
        'benchmark_final_value': bh_final_value,
        'benchmark_return_pct': bh_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'total_trades': len(trades),
        'trades': trades,
        'df_results': df_clean[['Close', 'Signal_Action', 'Portfolio_Value', 'Buy_Hold_Value']]
    }

def run_multi_trade_backtest(df_signals: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Simulates an active trading strategy with multiple trades (scale-in/scale-out).
    - Accumulates position by buying 25% of initial capital on BUY signals (max 4 times).
    - Reduces position by selling 50% of the current shares held on SELL signals.
    - Captures multiple trade events mimicking the market Close prices.
    """
    cash = initial_capital
    position = 0.0  # Number of shares held
    portfolio_values = []
    trades = []
    
    # We drop any rows with NaN in indicators first so we start where signals are valid
    df_clean = df_signals.dropna(subset=['SMA_20', 'RSI', 'MACD']).copy()
    
    if df_clean.empty:
        return {'error': 'No data available to backtest after removing NaNs.'}
        
    # Restrict to the most recent 30 calendar days of data
    end_date = df_clean.index[-1]
    start_trade_date = end_date - pd.Timedelta(days=30)
    df_clean = df_clean.loc[df_clean.index >= start_trade_date].copy()
    
    if df_clean.empty:
        return {'error': 'No recent data available for the last 30 days.'}
        
    buy_hold_shares = initial_capital / df_clean['Close'].iloc[0]
    tranche_size = initial_capital * 0.25
    
    for idx, row in df_clean.iterrows():
        close_price = row['Close']
        signal = row['Signal_Action']
        
        # 1. Process Signals
        if signal == 'BUY':
            # Check if we have enough cash to buy a 25% tranche
            if cash >= tranche_size:
                shares_bought = tranche_size / close_price
                position += shares_bought
                cash -= tranche_size
                trades.append({
                    'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                    'type': 'BUY (25% Tranche)',
                    'price': close_price,
                    'shares': shares_bought,
                    'portfolio_value': cash + (position * close_price)
                })
            elif cash > 10.0:  # Deploy remaining cash if it is small but valid
                shares_bought = cash / close_price
                position += shares_bought
                cash = 0.0
                trades.append({
                    'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                    'type': 'BUY (Remaining Cash)',
                    'price': close_price,
                    'shares': shares_bought,
                    'portfolio_value': position * close_price
                })
        elif signal == 'SELL' and position > 0.0:
            # Sell 50% of the current position
            shares_sold = position * 0.5
            cash_received = shares_sold * close_price
            position -= shares_sold
            cash += cash_received
            trades.append({
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'type': 'SELL (50% Position)',
                'price': close_price,
                'shares': shares_sold,
                'portfolio_value': cash + (position * close_price)
            })
            
        # 2. Track daily portfolio value
        current_val = cash + (position * close_price)
        portfolio_values.append(current_val)
        
    df_clean['Portfolio_Value'] = portfolio_values
    df_clean['Buy_Hold_Value'] = buy_hold_shares * df_clean['Close']
    
    # 3. Performance Metrics
    final_value = portfolio_values[-1]
    strategy_return = ((final_value - initial_capital) / initial_capital) * 100
    
    bh_final_value = df_clean['Buy_Hold_Value'].iloc[-1]
    bh_return = ((bh_final_value - initial_capital) / initial_capital) * 100
    
    # Daily returns
    df_clean['Daily_Return'] = df_clean['Portfolio_Value'].pct_change()
    mean_return = df_clean['Daily_Return'].mean()
    std_return = df_clean['Daily_Return'].std()
    
    if std_return > 0 and not pd.isna(std_return):
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        
    # Max Drawdown
    df_clean['Peak'] = df_clean['Portfolio_Value'].cummax()
    df_clean['Drawdown'] = (df_clean['Portfolio_Value'] - df_clean['Peak']) / df_clean['Peak']
    max_drawdown = df_clean['Drawdown'].min() * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'strategy_return_pct': strategy_return,
        'benchmark_final_value': bh_final_value,
        'benchmark_return_pct': bh_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'total_trades': len(trades),
        'trades': trades,
        'df_results': df_clean[['Close', 'Signal_Action', 'Portfolio_Value', 'Buy_Hold_Value']]
    }

if __name__ == '__main__':
    # Test Backtester
    from data_loader import fetch_stock_data
    from indicators import append_all_indicators
    from agent import generate_signals
    
    df = fetch_stock_data("AAPL", period="1y")
    df_ind = append_all_indicators(df)
    df_sig = generate_signals(df_ind, news_sentiment=0.1)
    results = run_backtest(df_sig)
    
    print("\nBACKTEST RESULTS FOR AAPL:")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Strategy Value: ${results['final_value']:.2f} ({results['strategy_return_pct']:.2f}%)")
    print(f"Final Benchmark Value: ${results['benchmark_final_value']:.2f} ({results['benchmark_return_pct']:.2f}%)")
    print(f"Annualized Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades Executed: {results['total_trades']}")
