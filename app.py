import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import yfinance as yf

# Adjust path to include the current folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    fetch_stock_data, 
    fetch_stock_news, 
    fetch_company_info, 
    fetch_industry_news,
    fetch_trending_tickers_with_ratings
)
from indicators import append_all_indicators
from sentiment import get_average_sentiment, analyze_headline_sentiment
from agent import generate_signals, get_latest_recommendation
from backtester import run_backtest, run_multi_trade_backtest
def get_ticker_difference_explanation(tickers: list) -> str:
    """
    Returns an educational explanation of the structural differences between selected tickers.
    """
    if len(tickers) < 2:
        return ""
    
    # Sort tickers to make comparisons order-independent
    sorted_tickers = sorted([t.upper() for t in tickers])
    
    # 1. Alphabet Class A vs C
    if 'GOOG' in sorted_tickers and 'GOOGL' in sorted_tickers:
        return """
**Alphabet Inc. (Google) Ticker Difference:**
* **GOOGL (Class A)**: Represents voting shares. Shareholders get **1 vote per share**.
* **GOOG (Class C)**: Represents non-voting shares. Shareholders get **no voting rights**.
* *Performance Note*: They trade at very similar prices, but GOOGL sometimes trades at a slight premium due to voting rights. Class B shares are held inside by founders/insiders and are not publicly traded.
"""
        
    # 2. Berkshire Hathaway Class A vs B
    has_brk_a = any(t in sorted_tickers for t in ('BRK-A', 'BRK.A'))
    has_brk_b = any(t in sorted_tickers for t in ('BRK-B', 'BRK.B'))
    if has_brk_a and has_brk_b:
        return """
**Berkshire Hathaway Ticker Difference:**
* **Class A (BRK-A)**: The original stock. It has never been split. It is one of the most expensive single shares in the world, comes with full voting rights, and can be converted into Class B shares.
* **Class B (BRK-B)**: A fractional share. It was introduced in 1996 to make the stock accessible to retail investors. One share of Class B represents **1/1500th** of a Class A share's value and carries **1/10,000th** of the voting rights.
"""
        
    # 3. NSE vs BSE Dual Listing (India)
    has_ns = any('.NS' in t for t in sorted_tickers)
    has_bo = any('.BO' in t for t in sorted_tickers)
    if has_ns and has_bo:
        # Check if they are listings of the same company (e.g. TCS.NS and TCS.BO)
        base_names = [t.split('.')[0] for t in sorted_tickers if ('.NS' in t or '.BO' in t)]
        if len(set(base_names)) == 1:
            company_base = base_names[0]
            return f"""
**NSE vs. BSE Dual Listing ({company_base}):**
* **{company_base}.NS**: Traded on the **National Stock Exchange of India (NSE)**. This is generally the larger exchange with higher trading volume and tighter spreads.
* **{company_base}.BO**: Traded on the **Bombay Stock Exchange (BSE)**. This is the oldest exchange in Asia and lists more companies.
* *Performance Note*: Arbitrage keeps prices extremely close, but minor volume and currency spreads can exist.
"""
            
    # 4. Cross-border listings (e.g. US vs European listing)
    # Check if we have a US listing (no suffix) vs a Frankfurt (.F), XETRA (.DE) or London (.L) listing
    us_tickers = [t for t in sorted_tickers if '.' not in t and '-' not in t]
    euro_tickers = [t for t in sorted_tickers if any(suffix in t for suffix in ('.F', '.DE', '.SG', '.MU', '.L'))]
    if us_tickers and euro_tickers:
        return """
**Cross-Border Listing (US vs. Europe):**
* **US Listing** (e.g., Nasdaq/NYSE): Traded in **USD** during US market hours. This is the primary listing where the company is headquartered, typically having the highest liquidity.
* **European Listings** (e.g., `.DE` for Xetra, `.F` for Frankfurt, `.L` for London): Traded in **EUR** or **GBP** during European market hours.
* *Performance Note*: They represent ownership of the same company, but their daily price charts differ due to exchange rates (USD vs. EUR) and differences in local trading hours.
"""
        
    return """
**General Ticker Differences:**
* **Suffixes (.NS, .BO, .L, .DE)**: Indicate the exchange where the stock is listed (e.g., NSE India, BSE India, London, Xetra Germany).
* **Share Classes (A, B, C)**: Represent different classes of ownership. Class A shares typically have voting rights, Class B may have different voting weights, and Class C shares are usually non-voting.
"""

def get_share_class_details(ticker: str, company_name: str = "") -> dict:
    """
    Determines the share class (Class A, Class B, Class C, ADR, Ordinary, etc.)
    and voting rights from the ticker and company name.
    """
    ticker_upper = ticker.upper().replace('.', '-')
    
    # Static mappings for famous ones
    if ticker_upper == 'GOOGL':
        return {
            'class': 'Class A Common Stock',
            'voting': '1 vote per share',
            'rights_desc': 'Voting shares representing ownership with full voting rights.',
            'siblings': ['GOOG']
        }
    if ticker_upper == 'GOOG':
        return {
            'class': 'Class C Capital Stock',
            'voting': 'No voting rights',
            'rights_desc': 'Non-voting shares representing ownership without voting rights.',
            'siblings': ['GOOGL']
        }
    if ticker_upper in ('BRK-A', 'BRK.A'):
        return {
            'class': 'Class A Common Stock',
            'voting': '1 vote per share (full voting)',
            'rights_desc': 'Original high-value shares. Can be converted into Class B shares, but not vice versa.',
            'siblings': ['BRK-B']
        }
    if ticker_upper in ('BRK-B', 'BRK.B'):
        return {
            'class': 'Class B Common Stock',
            'voting': '1/10,000th vote per share',
            'rights_desc': 'Fractional retail-accessible shares representing 1/1500th of a Class A share.',
            'siblings': ['BRK-A']
        }
    
    # Generic parsing based on name
    name_lower = company_name.lower()
    
    if ticker_upper == 'UAA':
        return {'class': 'Class A Common Stock', 'voting': '1 vote per share', 'rights_desc': 'Voting Class A shares.', 'siblings': ['UA']}
    if ticker_upper == 'UA':
        return {'class': 'Class C Common Stock', 'voting': 'No voting rights', 'rights_desc': 'Non-voting Class C shares.', 'siblings': ['UAA']}
    if ticker_upper == 'ZG':
        return {'class': 'Class A Common Stock', 'voting': '1 vote per share', 'rights_desc': 'Voting Class A shares.', 'siblings': ['Z']}
    if ticker_upper == 'Z':
        return {'class': 'Class C Common Stock', 'voting': 'No voting rights', 'rights_desc': 'Non-voting Class C shares.', 'siblings': ['ZG']}
    if ticker_upper == 'FOXA':
        return {'class': 'Class A Common Stock', 'voting': '1 vote per share', 'rights_desc': 'Voting Class A shares.', 'siblings': ['FOX']}
    if ticker_upper == 'FOX':
        return {'class': 'Class B Common Stock', 'voting': 'No voting / limited voting', 'rights_desc': 'Class B shares.', 'siblings': ['FOXA']}
    if ticker_upper == 'NWSA':
        return {'class': 'Class A Common Stock', 'voting': 'Limited voting', 'rights_desc': 'Class A shares.', 'siblings': ['NWS']}
    if ticker_upper == 'NWS':
        return {'class': 'Class B Common Stock', 'voting': 'Full voting', 'rights_desc': 'Class B shares.', 'siblings': ['NWSA']}

    # Fallback heuristic checks
    if 'class a' in name_lower or ticker_upper.endswith('A'):
        return {'class': 'Class A Stock', 'voting': 'Typically 1 vote per share', 'rights_desc': 'Voting share class.'}
    if 'class b' in name_lower or ticker_upper.endswith('B'):
        return {'class': 'Class B Stock', 'voting': 'Variable voting rights', 'rights_desc': 'Secondary share class (often super-voting or fractional).'}
    if 'class c' in name_lower or ticker_upper.endswith('C'):
        return {'class': 'Class C Stock', 'voting': 'Typically non-voting', 'rights_desc': 'Non-voting share class.'}
    if 'adr' in name_lower or 'sponsored adr' in name_lower:
        return {'class': 'American Depositary Receipt (ADR)', 'voting': 'Varies (dependent on custodian bank)', 'rights_desc': 'Represents shares of a foreign company trading on a US exchange.'}
    
    return {
        'class': 'Common Stock / Ordinary Shares',
        'voting': '1 vote per share (standard)',
        'rights_desc': 'Standard voting equity shares representing standard ownership.'
    }

def expand_search_results(quotes: list, search_query: str) -> list:
    """
    Expands search results to make sure sibling share classes (A, B, C)
    are included if one of them is in the search results or query.
    """
    expanded = list(quotes)
    existing_symbols = {q.get('symbol', '').upper() for q in quotes}
    
    # Known sibling pairs/groups
    sibling_groups = [
        {'GOOG', 'GOOGL'},
        {'BRK-A', 'BRK-B'},
        {'UAA', 'UA'},
        {'ZG', 'Z'},
        {'FOX', 'FOXA'},
        {'NWSA', 'NWS'}
    ]
    
    # Special query-based injection first
    q_lower = search_query.lower().strip()
    if 'google' in q_lower or 'alphabet' in q_lower:
        for sym in ['GOOGL', 'GOOG']:
            if sym not in existing_symbols:
                expanded.append({
                    'symbol': sym,
                    'shortname': 'Alphabet Inc. Class A',
                    'longname': 'Alphabet Inc. Class A Common Stock',
                    'exchDisp': 'NASDAQ',
                    'exchange': 'NMS',
                    'quoteType': 'EQUITY',
                    'typeDisp': 'Equity',
                    'sector': 'Technology',
                    'industry': 'Internet Content & Information'
                } if sym == 'GOOGL' else {
                    'symbol': sym,
                    'shortname': 'Alphabet Inc. Class C',
                    'longname': 'Alphabet Inc. Class C Capital Stock',
                    'exchDisp': 'NASDAQ',
                    'exchange': 'NMS',
                    'quoteType': 'EQUITY',
                    'typeDisp': 'Equity',
                    'sector': 'Technology',
                    'industry': 'Internet Content & Information'
                })
                existing_symbols.add(sym)
    elif 'berkshire' in q_lower or 'brk' in q_lower:
        for sym in ['BRK-A', 'BRK-B']:
            if sym not in existing_symbols:
                expanded.append({
                    'symbol': sym,
                    'shortname': 'Berkshire Hathaway Class A',
                    'longname': 'Berkshire Hathaway Inc. Class A',
                    'exchDisp': 'NYSE',
                    'exchange': 'NYQ',
                    'quoteType': 'EQUITY',
                    'typeDisp': 'Equity',
                    'sector': 'Financial Services',
                    'industry': 'Insurance - Diversified'
                } if sym == 'BRK-A' else {
                    'symbol': sym,
                    'shortname': 'Berkshire Hathaway Class B',
                    'longname': 'Berkshire Hathaway Inc. Class B',
                    'exchDisp': 'NYSE',
                    'exchange': 'NYQ',
                    'quoteType': 'EQUITY',
                    'typeDisp': 'Equity',
                    'sector': 'Financial Services',
                    'industry': 'Insurance - Diversified'
                })
                existing_symbols.add(sym)
                
    # Search quotes and add siblings if any quotes match
    for q in quotes:
        sym = q.get('symbol', '').upper()
        for group in sibling_groups:
            # check with/without dot notation
            normalized_sym = sym.replace('.', '-')
            if sym in group or normalized_sym in group:
                for sibling in group:
                    sibling_norm = sibling.replace('-', '.')
                    sibling_dash = sibling.replace('.', '-')
                    if sibling not in existing_symbols and sibling_norm not in existing_symbols and sibling_dash not in existing_symbols:
                        new_quote = q.copy()
                        new_quote['symbol'] = sibling
                        if sibling == 'GOOGL':
                            new_quote['shortname'] = 'Alphabet Inc. Class A'
                            new_quote['longname'] = 'Alphabet Inc. Class A Common Stock'
                        elif sibling == 'GOOG':
                            new_quote['shortname'] = 'Alphabet Inc. Class C'
                            new_quote['longname'] = 'Alphabet Inc. Class C Capital Stock'
                        elif sibling == 'BRK-A':
                            new_quote['shortname'] = 'Berkshire Hathaway Class A'
                            new_quote['longname'] = 'Berkshire Hathaway Inc. Class A'
                        elif sibling == 'BRK-B':
                            new_quote['shortname'] = 'Berkshire Hathaway Class B'
                            new_quote['longname'] = 'Berkshire Hathaway Inc. Class B'
                        else:
                            new_quote['shortname'] = f"{q.get('shortname', '')} ({sibling})"
                            new_quote['longname'] = f"{q.get('longname', '')} ({sibling})"
                        expanded.append(new_quote)
                        existing_symbols.add(sibling)
                        existing_symbols.add(sibling_norm)
                        existing_symbols.add(sibling_dash)
                        
    return expanded

def get_fetch_period(period: str) -> str:
    """
    Returns the minimum period to fetch from yfinance to ensure we have enough
    historical data to calculate indicators (e.g. at least 1 year).
    """
    if period in ["3d", "1wk", "1mo", "3mo", "6mo", "1y"]:
        return "1y"
    elif period == "2y":
        return "2y"
    elif period == "5y":
        return "5y"
    elif period == "10y":
        return "10y"
    elif period == "max":
        return "max"
    return "1y"

def slice_df_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Slices a DataFrame to the requested period based on its index.
    """
    if df.empty:
        return df
    
    end_date = df.index[-1]
    if period == "3d":
        return df.tail(3)
    elif period == "1wk":
        start_date = end_date - pd.Timedelta(days=7)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 5:
            return df.tail(5)
        return sliced_df
    elif period == "1mo":
        start_date = end_date - pd.Timedelta(days=30)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 20:
            return df.tail(20)
        return sliced_df
    elif period == "3mo":
        start_date = end_date - pd.Timedelta(days=91)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 60:
            return df.tail(60)
        return sliced_df
    elif period == "6mo":
        start_date = end_date - pd.Timedelta(days=182)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 120:
            return df.tail(120)
        return sliced_df
    elif period == "1y":
        start_date = end_date - pd.Timedelta(days=365)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 250:
            return df.tail(250)
        return sliced_df
    elif period == "5y":
        start_date = end_date - pd.Timedelta(days=365 * 5 + 1)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 250 * 5:
            return df.tail(250 * 5)
        return sliced_df
    elif period == "10y":
        start_date = end_date - pd.Timedelta(days=365 * 10 + 2)
        sliced_df = df.loc[df.index >= start_date]
        if len(sliced_df) < 250 * 10:
            return df.tail(250 * 10)
        return sliced_df
    return df

def run_agent_analysis_for_comparison(ticker_symbol: str, period: str) -> dict:
    """
    Runs a fast technical and sentiment analysis on a comparison stock ticker.
    """
    ticker_symbol = ticker_symbol.upper().strip()
    fetch_p = get_fetch_period(period)
    
    # Resolve simple dot notation to hyphen if yfinance returns empty
    if '.' in ticker_symbol:
        test_df = cached_fetch_stock_data(ticker_symbol, period=fetch_p)
        if test_df.empty:
            ticker_symbol = ticker_symbol.replace('.', '-')
            
    try:
        df_raw = cached_fetch_stock_data(ticker_symbol, period=fetch_p)
        if df_raw.empty:
            # Try Indian market suffix fallback if not appended
            if not ticker_symbol.endswith(('.NS', '.BO')):
                test_ticker = f"{ticker_symbol}.NS"
                test_df = cached_fetch_stock_data(test_ticker, period=fetch_p)
                if not test_df.empty:
                    ticker_symbol = test_ticker
                    df_raw = test_df
            
            if df_raw.empty:
                return {'error': f"No data found for {ticker_symbol}"}
        
        info = cached_fetch_company_info(ticker_symbol)
        news = cached_fetch_stock_news(ticker_symbol)
        
        sentiment = get_average_sentiment(news)
        df_ind = append_all_indicators(df_raw)
        df_sig = generate_signals(df_ind, news_sentiment=sentiment)
        
        # Slice both df_raw and df_sig to the requested period after indicators/signals generation
        df_raw = slice_df_to_period(df_raw, period)
        df_sig = slice_df_to_period(df_sig, period)
        
        latest_row = df_sig.iloc[-1]
        
        # Calculate returns
        start_price = df_raw['Close'].iloc[0]
        end_price = df_raw['Close'].iloc[-1]
        period_return = ((end_price - start_price) / start_price) * 100
        
        # Volatility (annualized)
        daily_returns = df_raw['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if not daily_returns.empty else 0.0
        
        # Max Drawdown
        roll_max = df_raw['Close'].cummax()
        drawdowns = (df_raw['Close'] - roll_max) / roll_max
        max_drawdown = drawdowns.min() * 100
        
        share_info = get_share_class_details(ticker_symbol, info.get('name', ''))
        
        return {
            'ticker': ticker_symbol,
            'name': info.get('name', ticker_symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'price': end_price,
            'prev_price': df_raw['Close'].iloc[-2] if len(df_raw) > 1 else end_price,
            'change_pct': ((end_price - df_raw['Close'].iloc[-2])/df_raw['Close'].iloc[-2])*100 if len(df_raw) > 1 else 0.0,
            'period_return': period_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'rsi': latest_row.get('RSI', 50.0),
            'macd_hist': latest_row.get('MACD_Hist', 0.0),
            'trend': "UP" if latest_row.get('Close', 0.0) > latest_row.get('SMA_20', 0.0) else "DOWN",
            'sentiment': sentiment,
            'score': latest_row.get('Agent_Score', 0.0),
            'action': latest_row.get('Signal_Action', 'HOLD'),
            'share_class': share_info['class'],
            'voting': share_info['voting'],
            'history': df_raw
        }
    except Exception as e:
        return {'error': f"Error parsing {ticker_symbol}: {str(e)}"}

        return {'error': f"Error parsing {ticker_symbol}: {str(e)}"}

def call_provider_llm_api(provider: str, model: str, prompt: str, api_key: str, ollama_url: str = "http://localhost:11434", max_tokens: int = 1500) -> str:
    import urllib.request
    import json
    
    provider = provider.lower()
    
    if provider == "google gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens
            }
        }
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
    elif provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
    elif provider == "anthropic claude":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    elif provider == "local ollama":
        url = f"{ollama_url.rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens
            }
        }
    elif provider == "ollama cloud":
        url = "https://ollama.com/api/chat"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens
            }
        }
    elif provider == "llama api (cloud/hosted)":
        url = ollama_url.strip()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        if "/api/chat" in url:
            # Ollama hosted format
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens
                }
            }
        else:
            # Standard OpenAI compatible format
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens
            }
    else:
        return f"Error: Unknown provider '{provider}'"
        
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            if provider == "google gemini":
                text = res_json['candidates'][0]['content']['parts'][0]['text']
            elif provider in ("openai", "groq"):
                text = res_json['choices'][0]['message']['content']
            elif provider == "anthropic claude":
                text = res_json['content'][0]['text']
            elif provider in ("local ollama", "ollama cloud"):
                text = res_json['message']['content']
            elif provider == "llama api (cloud/hosted)":
                if "/api/chat" in url:
                    text = res_json['message']['content']
                else:
                    text = res_json['choices'][0]['message']['content']
            else:
                text = ""
                
            if text:
                # Clean up LaTeX/math tags that confuse the front-end markdown renderer
                # Replace backslashed brackets/parentheses and double dollars with plain text
                text = text.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
                text = text.replace("$$", "").replace("$", "")
                return text
            return "Error parsing response"
    except Exception as e:
        err_msg = str(e)
        if "10061" in err_msg or "refused" in err_msg:
            if provider == "local ollama":
                return f"❌ **Connection Refused**: Could not connect to the local Ollama server at `{url}`.\n\n" \
                       f"Please make sure your local Ollama app is open and running on your computer."
            elif provider == "ollama cloud":
                return f"❌ **Connection Refused**: Could not connect to the Ollama Cloud server at `{url}`.\n\n" \
                       f"Please verify your API key and network connection."
            else:
                return f"❌ **Connection Refused**: Could not connect to the Llama server at `{url}`.\n\n" \
                       f"Please verify your **Llama API Endpoint URL** and ensure it is correct."
def get_llm_qa_response(question: str, tab_context: str, ticker: str, stock_stats: dict, provider: str, model: str, api_key: str, ollama_url: str) -> str:
    # Base prompt for response generation
    base_prompt = f"""
You are AlphaAgent, a senior quantitative financial analyst.

Context:
- Current Tab / View: {tab_context}
- Stock Ticker: {ticker}
- Company Name: {stock_stats.get('name', ticker)}
- Market Price: ${stock_stats.get('price', 0.0):,.2f} ({stock_stats.get('change_pct', 0.0):+.2f}%)
- Technical indicators: RSI (14) = {stock_stats.get('rsi', 50.0):.1f}, MACD Histogram = {stock_stats.get('macd_hist', 0.0):.4f}
- VADER news sentiment (last 15 headlines): {stock_stats.get('sentiment', 0.0):+.2f}
- Recent Headlines:
{stock_stats.get('news_summary', 'No headlines available')}

User Query: {question}

Instructions:
1. Provide a complete, fully formed, and highly understandable answer.
2. Translate complex financial terms, numbers, or logic into simple concepts using clear, intuitive analogies (e.g. explain like explaining to a 5-year-old, such as comparing RSI momentum to a runner slowing down, or Bollinger Bands to safety boundaries), but express them in a mature, professional language appropriate for adults.
3. Base all details strictly on the provided context and verified financial facts; avoid speculative guesses.
4. Structure & Formatting Requirements:
   - Output in clean, highly structured Markdown.
   - Use bold subheadings (e.g., ### Section Name) for logical sections.
   - Use bullet points with bold key metrics.
   - Separate distinct ideas with spacing or horizontal rules (`---`).
   - Do not output any HTML tags.
   - Start directly with the structured answer (no conversational preamble).
   - IMPORTANT: DO NOT write any math symbols or equations using LaTeX delimiters like \\[, \\], \\(, \\), or double/single dollar signs ($). Write all mathematical equations and formulas in simple, plain english text representation (e.g., use "Sharpe Ratio = Mean Daily Return / Volatility" instead of LaTeX notation).
"""

    # Generate response directly with a generous max_tokens limit (e.g., 4000 tokens)
    response = call_provider_llm_api(provider, model, base_prompt, api_key, ollama_url, max_tokens=4000)
    return response

def get_local_qa_response(question: str, tab_id: str, ticker: str, current_price: float, rsi: float, macd_hist: float, sentiment: float, rsi_oversold: float, rsi_overbought: float) -> str:
    q = question.lower()
    
    if "rsi" in q:
        return f"""
**RSI (Relative Strength Index) Explanation:**
* The current RSI for **{ticker}** is **{rsi:.2f}**.
* **What it means:** RSI is a momentum indicator that ranges from 0 to 100. 
  - A reading below {rsi_oversold} indicates the stock is **oversold** (historically undervalued/bullish signal).
  - A reading above {rsi_overbought} indicates the stock is **overbought** (historically overvalued/bearish signal).
  - Since the current RSI is **{rsi:.2f}**, the stock is in **{"Oversold (Bullish)" if rsi < rsi_oversold else ("Overbought (Bearish)" if rsi > rsi_overbought else "Neutral")}** territory.
"""
    elif "macd" in q:
        return f"""
**MACD (Moving Average Convergence Divergence) Explanation:**
* The current MACD histogram value for **{ticker}** is **{macd_hist:.4f}**.
* **What it means:** MACD shows the relationship between two moving averages (12-day and 26-day EMA).
  - When the MACD line crosses *above* the signal line (Histogram becomes positive, current: {macd_hist:+.4f}), it indicates a **Bullish Crossover** (momentum is accelerating upward).
  - When it crosses *below* the signal line (Histogram becomes negative), it indicates a **Bearish Crossover** (momentum is accelerating downward).
"""
    elif "bollinger" in q or "bb" in q:
        return """
**Bollinger Bands Explanation:**
* Bollinger Bands consist of a middle 20-day Simple Moving Average (SMA) and two outer bands placed 2 standard deviations away.
* **Interpretation:**
  - When the stock price touches or exceeds the **Upper Band**, it is considered high relative to its recent average (potentially overbought/volatile).
  - When the stock price touches or falls below the **Lower Band**, it is considered low relative to its recent average (potentially oversold/volatile).
"""
    elif "sentiment" in q or "vader" in q or "news" in q:
        return f"""
**Sentiment Analysis Explanation:**
* The average company sentiment score for **{ticker}** is **{sentiment:+.2f}**.
* **What it means:** We analyze financial news headlines using the VADER sentiment engine.
  - Scores range from `-1.0` (highly bearish) to `+1.0` (highly bullish).
  - A score above `+0.15` indicates optimistic news, while below `-0.15` indicates pessimistic news.
  - Currently, {ticker}'s sentiment is **{"Bullish" if sentiment > 0.15 else ("Bearish" if sentiment < -0.15 else "Neutral")}**.
"""
    elif "drawdown" in q:
        return """
**Maximum Drawdown Explanation:**
* **What it means:** Max Drawdown measures the largest historical peak-to-trough decline in the stock price (or portfolio value) during the selected period.
* It is a key measure of risk, showing the worst-case loss scenario an investor would have faced if they bought at the absolute peak and sold at the absolute bottom.
"""
    elif "capital" in q or "backtest" in q:
        return """
**Backtest Metrics Explanation:**
* Our backtester simulates buying and selling the stock historically using the AI Agent's signals.
* **Benchmark (Buy & Hold):** Represents the return you would get if you simply bought the stock on Day 1 and held it until the final day.
* **Strategy Return:** Represents the return generated by executing the Agent's BUY and SELL signals. If Strategy Return exceeds the Benchmark, the AI strategy successfully beat the market.
"""
    
    return f"""
**Local Stock Analyst Response:**
* I analyzed your question regarding **{ticker}** under the **{tab_id.upper()}** view.
* **Current Stock Price:** ${current_price:,.2f}
* **RSI (14):** {rsi:.1f}
* **News Sentiment:** {sentiment:+.2f}
* *Note: To enable full natural language chat powered by Gemini 1.5 Flash with knowledge of global news and real-world occurrences, please enter your Gemini API Key in the sidebar.*
"""

def render_ai_assistant_widget(tab_id: str, tab_context: str, ticker: str, stock_stats: dict, api_key: str, rsi_oversold: float, rsi_overbought: float):
    # Read settings from session state
    provider = st.session_state.get("llm_provider", "Google Gemini")
    model = st.session_state.get("selected_model", "gemini-1.5-flash")
    ollama_url = st.session_state.get("ollama_url", "")
    
    is_active = provider.lower() != "local expert"
    
    st.markdown("---")
    
    # Header with toggle info
    col_hdr_title, col_hdr_badge = st.columns([4, 2])
    with col_hdr_title:
        st.markdown("### 💬 Ask AlphaAgent AI Assistant")
    with col_hdr_badge:
        if is_active:
            st.markdown(f"<span style='float:right; padding: 4px 10px; background: rgba(0, 176, 155, 0.1); border-radius: 6px; font-size: 0.82rem; color: #00b09b; border: 1px solid rgba(0,176,155,0.2);'>{provider} ({model}) Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='float:right; padding: 4px 10px; background: rgba(247, 151, 30, 0.1); border-radius: 6px; font-size: 0.82rem; color: #f7971e; border: 1px solid rgba(247,151,30,0.2);'>Local Expert Fallback Active</span>", unsafe_allow_html=True)

    # Initialize history for this tab
    history_key = f"chat_history_{tab_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
        
    # Render chat history
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Form for chat input
    with st.form(key=f"chat_form_{tab_id}", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            q = st.text_input("Ask about terms, graphs, or trends in this tab:", key=f"chat_input_val_{tab_id}", placeholder="e.g. Explain MACD crossover? or How to interpret the relative returns chart?")
        with col_btn:
            submit = st.form_submit_button("Ask 🚀")
            
        if submit and q:
            # Append user message
            st.session_state[history_key].append({"role": "user", "content": q})
            
            # Generate response
            with st.spinner("AlphaAgent is thinking..."):
                if is_active:
                    response = get_llm_qa_response(q, tab_context, ticker, stock_stats, provider, model, api_key, ollama_url)
                else:
                    response = get_local_qa_response(
                        q, tab_id, ticker, 
                        stock_stats.get('price', 0.0), 
                        stock_stats.get('rsi', 50.0), 
                        stock_stats.get('macd_hist', 0.0), 
                        stock_stats.get('sentiment', 0.0),
                        rsi_oversold, rsi_overbought
                    )
                st.session_state[history_key].append({"role": "assistant", "content": response})
            st.rerun()

# --- STREAMLIT CACHING WRAPPERS ---

# --- STREAMLIT CACHING WRAPPERS ---
@st.cache_data(ttl=3600)  # Cache trending tickers for 1 hour
def cached_fetch_trending_tickers():
    return fetch_trending_tickers_with_ratings()

@st.cache_data(ttl=3600)  # Cache company profile for 1 hour
def cached_fetch_company_info(ticker_symbol: str) -> dict:
    return fetch_company_info(ticker_symbol)

@st.cache_data(ttl=1800)  # Cache price history for 30 minutes
def cached_fetch_stock_data(ticker_symbol: str, period: str) -> pd.DataFrame:
    return fetch_stock_data(ticker_symbol, period=period)

@st.cache_data(ttl=10)  # Cache ticker news for 10 seconds
def cached_fetch_stock_news(ticker_symbol: str) -> list:
    return fetch_stock_news(ticker_symbol)

@st.cache_data(ttl=10)  # Cache industry trends news for 10 seconds
def cached_fetch_industry_news(search_query: str) -> list:
    return fetch_industry_news(search_query)

@st.cache_data(ttl=3600)  # Cache ticker search results for 1 hour
def cached_search_ticker(query: str) -> list:
    if not query:
        return []
    import urllib.request
    import urllib.parse
    import json
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        data = json.loads(response.read().decode())
        quotes = data.get('quotes', [])
        # Filter for equities and ETFs only
        filtered_quotes = [q for q in quotes if q.get('quoteType') in ('EQUITY', 'ETF')]
        return filtered_quotes
    except Exception as e:
        print(f"Error searching tickers: {e}")
        return []


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AlphaAgent | AI Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main body background & font family */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant gradients and header customization */
    .header-container {
        background: linear-gradient(135deg, #1e1e3f 0%, #111122 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #8f9cae;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Metrics and glassmorphism styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Signal Badges */
    .badge {
        padding: 0.5rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .badge-buy {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
    }
    .badge-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .badge-hold {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
    }
    
    /* News Sentiment Styling */
    .news-card {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.02);
        border-color: rgba(0, 242, 254, 0.2);
    }
    
    /* Pulsating Green Dot and Badge */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 176, 155, 0.1);
        border: 1px solid rgba(0, 176, 155, 0.3);
        border-radius: 8px;
        padding: 0.25rem 0.6rem;
        color: #00b09b;
        font-size: 0.85rem;
        font-weight: 600;
        vertical-align: middle;
        margin-left: 0.8rem;
    }
    .pulsing-dot {
        width: 8px;
        height: 8px;
        background: #00b09b;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(0, 176, 155, 0.7);
        animation: pulse-green 1.5s infinite;
    }
    @keyframes pulse-green {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 176, 155, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(0, 176, 155, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 176, 155, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='font-weight:800; background: linear-gradient(to right, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>⚙️ Agent Settings</h2>", unsafe_allow_html=True)

# Initialize recent searches list
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = ["AAPL", "TSLA", "NVDA"]

# Stock inputs
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "AAPL"

ticker = st.sidebar.text_input(
    "Enter Stock Ticker Symbol", 
    value=st.session_state.ticker_input,
    help="For international tickers, add the exchange suffix (e.g., TCS.NS for Tata Consultancy Services on the National Stock Exchange of India, or TCS.BO for BSE)."
).upper().strip()

# Sync state
st.session_state.ticker_input = ticker

# If active ticker is valid and not already in recent searches, push it to history
if ticker and ticker not in st.session_state.recent_searches:
    st.session_state.recent_searches.insert(0, ticker)
    # Limit to top 5 items
    st.session_state.recent_searches = st.session_state.recent_searches[:5]

# Render recent searches as pill suggestions
if st.session_state.recent_searches:
    st.sidebar.caption("🕒 Recent Searches (Click to Suggest):")
    recent_cols = st.sidebar.columns(3)
    for idx, recent_sym in enumerate(st.session_state.recent_searches):
        col_idx = idx % 3
        with recent_cols[col_idx]:
            if st.button(recent_sym, key=f"recent_{recent_sym}_{idx}", use_container_width=True):
                st.session_state.ticker_input = recent_sym
                st.rerun()

# Fetch company info early to get default industry search query
try:
    comp_info = cached_fetch_company_info(ticker)
except Exception:
    comp_info = {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown', 'summary': ''}

st.sidebar.markdown(f"**Analyzing:** {comp_info['name']}")
if comp_info['sector'] != 'Unknown':
    st.sidebar.caption(f"Sector: {comp_info['sector']} | Industry: {comp_info['industry']}")

time_period = st.sidebar.selectbox("Analysis Time Window", ["3d", "1wk", "1mo", "3mo", "1y", "5y", "10y", "max"], index=4)
starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=100.0, value=10000.0, step=100.0)
live_streaming = st.sidebar.toggle("Real-Time Data Streaming", value=True, help="Enable continuously fluctuating and updating real-time data for the current date.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-weight:600;'>🧠 Strategy Hyperparameters</h3>", unsafe_allow_html=True)

# Weight adjustments
st.sidebar.caption("Adjust Agent decision weights:")
w_tech = st.sidebar.slider("Technical Indicators (Quant)", 0.0, 1.0, 0.50, 0.05)
w_comp = st.sidebar.slider("Company Specific News", 0.0, 1.0, 0.30, 0.05)
w_ind = st.sidebar.slider("Industry & Sector News", 0.0, 1.0, 0.20, 0.05)

# Normalise weights
total_weight = w_tech + w_comp + w_ind
if total_weight > 0:
    w_tech_norm = w_tech / total_weight
    w_comp_norm = w_comp / total_weight
    w_ind_norm = w_ind / total_weight
else:
    w_tech_norm, w_comp_norm, w_ind_norm = 0.34, 0.33, 0.33
    
st.sidebar.caption(f"Normalized: Quant {w_tech_norm:.0%}, Comp News {w_comp_norm:.0%}, Industry {w_ind_norm:.0%}")

# Technical parameters
st.sidebar.markdown("<h4 style='font-weight:600;'>Technical Thresholds</h4>", unsafe_allow_html=True)
rsi_oversold = st.sidebar.slider("RSI Oversold Level (Buy)", 10, 40, 30)
rsi_overbought = st.sidebar.slider("RSI Overbought Level (Sell)", 60, 90, 70)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-weight:600;'>🔍 Search Parameters</h3>", unsafe_allow_html=True)
default_query = f"{comp_info['sector']} {comp_info['industry']} trends" if comp_info['sector'] != 'Unknown' else f"{comp_info['name']} competitors news"
industry_query = st.sidebar.text_input("Industry News Search Query", default_query, help="Modify this query to fetch relevant macroeconomic/sector news headlines from Google News.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-weight:600;'>🧠 AI Assistant Settings</h3>", unsafe_allow_html=True)

# Helper function to detect provider, model, and ollama_url
def detect_provider_and_model(api_key: str):
    api_key = api_key.strip()
    if not api_key:
        return "local expert", "", ""
    if api_key.startswith("AIzaSy"):
        return "Google Gemini", "gemini-1.5-flash", ""
    elif api_key.startswith("gsk_"):
        return "Groq", "llama-3.3-70b-versatile", ""
    elif api_key.startswith("sk-ant-"):
        return "Anthropic Claude", "claude-3-5-sonnet-20241022", ""
    elif api_key.startswith("sk-"):
        return "OpenAI", "gpt-4o-mini", ""
    else:
        return "Ollama Cloud", "gpt-oss:120b", "https://ollama.com"

# Retrieve env keys based on priority
api_key_env = os.environ.get("OLLAMA_API_KEY", 
              os.environ.get("GEMINI_API_KEY", 
              os.environ.get("GOOGLE_API_KEY", 
              os.environ.get("OPENAI_API_KEY", 
              os.environ.get("GROQ_API_KEY", 
              os.environ.get("ANTHROPIC_API_KEY", ""))))))

gemini_api_key = st.sidebar.text_input(
    "API Key",
    value=api_key_env,
    type="password",
    key="llm_api_key",
    help="Paste your Ollama Cloud, Gemini, OpenAI, Groq, or Anthropic Claude API Key here. The system will automatically detect the provider."
)

llm_provider, selected_model, ollama_url = detect_provider_and_model(gemini_api_key)
is_active = llm_provider.lower() != "local expert"

# Sync session state
st.session_state["llm_provider"] = llm_provider
st.session_state["selected_model"] = selected_model
st.session_state["ollama_url"] = ollama_url

# --- WEEKLY TRENDING TICKERS SIDEBAR EXPANDER ---
st.sidebar.markdown("---")
with st.sidebar.expander("🔥 Weekly Trending Tickers", expanded=True):
    st.caption("Weekly search interest score & news breakouts:")
    trending_list = cached_fetch_trending_tickers()
    if trending_list:
        for idx, item in enumerate(trending_list):
            rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            emoji = rank_emojis[idx] if idx < len(rank_emojis) else "📈"
            
            # Colored left border indicator based on score
            score_color = "#00b09b" if item['score'] >= 85 else ("#ffd200" if item['score'] >= 75 else "#8f9cae")
            
            # Stunning premium card layout
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 12px; margin-bottom: 5px; border-left: 4px solid {score_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.05rem; font-weight: 800; color: #ffffff;">{emoji} {item['symbol']}</span>
                    <span style="background: rgba(0, 242, 254, 0.1); border: 1px solid rgba(0, 242, 254, 0.3); padding: 2px 8px; border-radius: 8px; font-size: 0.8rem; font-weight: 700; color: #00f2fe;">🔥 {item['score']}%</span>
                </div>
                <div style="font-size: 0.78rem; color: #8f9cae; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {item['name']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis action button
            if st.button(f"Analyze {item['symbol']}", key=f"trend_btn_{item['symbol']}_{idx}", use_container_width=True):
                st.session_state.ticker_input = item['symbol']
                st.rerun()
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    else:
        st.write("No trending data available.")

# --- DISCLAIMER / WARNING ---
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Disclaimer**: This app does not claim or guarantee assured financial results. "
    "Do not rely on the quantitative analysis, signals, sentiment scores, or AI consensus recommendations "
    "completely. Investing involves high risk. Always make your own independent investment decisions."
)

# ----------------- HEADER -----------------
live_badge_html = '<span class="live-badge"><span class="pulsing-dot"></span>LIVE STREAMING</span>' if live_streaming else ''

st.markdown(f"""
<div class="header-container">
    <h1 class="header-title" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <span>AlphaAgent AI Stock Analyst</span>
        {live_badge_html}
    </h1>
    <p class="header-subtitle">Real-Time Decision Intelligence Engine parsing technical indicators and financial market sentiment for <b>{ticker}</b></p>
</div>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
try:
    with st.spinner("Fetching data and running AI agents..."):
        df_raw = cached_fetch_stock_data(ticker, period=get_fetch_period(time_period))
        
        # 1. Fallback for Class Shares (e.g. BRK.A -> BRK-A)
        if df_raw.empty and '.' in ticker:
            fallback_ticker = ticker.replace('.', '-')
            df_fallback = cached_fetch_stock_data(fallback_ticker, period=get_fetch_period(time_period))
            if not df_fallback.empty:
                df_raw = df_fallback
                ticker = fallback_ticker
                try:
                    comp_info = cached_fetch_company_info(ticker)
                except Exception:
                    pass
                st.sidebar.info(f"💡 Redirected to **{ticker}** (Yahoo Finance format).")
        
        # 2. Smart fallback for Indian markets (e.g. TCS -> TCS.NS)
        if df_raw.empty and not ticker.endswith(('.NS', '.BO')):
            fallback_ticker = f"{ticker}.NS"
            df_fallback = cached_fetch_stock_data(fallback_ticker, period=get_fetch_period(time_period))
            if not df_fallback.empty:
                df_raw = df_fallback
                ticker = fallback_ticker
                try:
                    comp_info = cached_fetch_company_info(ticker)
                except Exception:
                    pass
                st.sidebar.info(f"💡 Redirected to **{ticker}** (NSE India).")
        
        if df_raw.empty:
            st.error(f"Error: No historical stock data returned for '{ticker}'. Please check if the ticker symbol is correct. For Indian stocks, try appending '.NS' (e.g., TCS.NS).")
            st.stop()
            
        # --- Real-Time Live Data Streaming (June 21, 2026 Reference) ---
        if live_streaming:
            rt_key = f"df_realtime_{ticker}_{time_period}"
            today_ts = pd.Timestamp("2026-06-21")
            
            if rt_key not in st.session_state:
                # Initialize realtime dataframe with historical baseline
                df_base = df_raw.copy()
                df_base.index = pd.to_datetime(df_base.index)
                
                # Make timezone-naive to avoid timezone comparison errors
                if df_base.index.tz is not None:
                    df_base.index = df_base.index.tz_localize(None)
                
                # Overwrite/Drop today's entry if it exists in base to start clean
                if today_ts in df_base.index:
                    df_base = df_base.drop(today_ts)
                
                # Try fetching live market data for initialization
                try:
                    ticker_obj = yf.Ticker(ticker)
                    live_data = ticker_obj.fast_info
                    init_close = live_data['lastPrice']
                    init_open = live_data.get('open', init_close)
                    init_high = live_data.get('dayHigh', init_close)
                    init_low = live_data.get('dayLow', init_close)
                    init_volume = live_data.get('lastVolume', 0)
                except Exception:
                    # Fallback to last row close
                    last_row = df_base.iloc[-1]
                    init_close = last_row["Close"]
                    init_open = last_row["Close"]
                    init_high = last_row["Close"]
                    init_low = last_row["Close"]
                    init_volume = last_row["Volume"]
                
                # Append today's candle
                new_row = pd.DataFrame({
                    "Open": init_open,
                    "High": init_high,
                    "Low": init_low,
                    "Close": init_close,
                    "Volume": init_volume
                }, index=[today_ts])
                new_row.index = pd.to_datetime(new_row.index)
                
                df_base = pd.concat([df_base, new_row])
                st.session_state[rt_key] = df_base
            else:
                # Fetch actual live data from the market
                df_base = st.session_state[rt_key]
                last_idx = df_base.index[-1]
                
                try:
                    ticker_obj = yf.Ticker(ticker)
                    live_data = ticker_obj.fast_info
                    new_close = live_data['lastPrice']
                    new_open = live_data.get('open', new_close)
                    new_high = live_data.get('dayHigh', new_close)
                    new_low = live_data.get('dayLow', new_close)
                    new_volume = live_data.get('lastVolume', 0)
                    
                    df_base.loc[last_idx, "Close"] = new_close
                    df_base.loc[last_idx, "Open"] = new_open
                    df_base.loc[last_idx, "High"] = new_high
                    df_base.loc[last_idx, "Low"] = new_low
                    df_base.loc[last_idx, "Volume"] = new_volume
                except Exception:
                    # Maintain existing values if fetch fails temporarily
                    pass
                    
                st.session_state[rt_key] = df_base
                
            df_raw = st.session_state[rt_key]
            
        # Fetch company news and industry trends
        news_items = cached_fetch_stock_news(ticker)
        ind_news_items = cached_fetch_industry_news(industry_query)
        
        # Sentiment Analysis
        avg_sentiment = get_average_sentiment(news_items)
        avg_industry_sentiment = get_average_sentiment(ind_news_items)
        
        # Calculate Technical Indicators
        df_indicators = append_all_indicators(df_raw)
        
        # Generate base signals
        df_signals = generate_signals(df_indicators, news_sentiment=avg_sentiment)
        
        # Adjust calculations dynamically according to parameters
        # Custom RSI limits
        df_signals['RSI_Signal'] = 0.0
        df_signals.loc[df_signals['RSI'] < rsi_oversold, 'RSI_Signal'] = 1.0
        df_signals.loc[df_signals['RSI'] > rsi_overbought, 'RSI_Signal'] = -1.0
        
        # Populate Industry News Sentiment Signal
        df_signals['Industry_Signal'] = 0.0
        if avg_industry_sentiment > 0.15:
            df_signals['Industry_Signal'] = 1.0
        elif avg_industry_sentiment < -0.15:
            df_signals['Industry_Signal'] = -1.0
            
        # Custom Weighting (Quant + Company News + Industry News)
        df_signals['Agent_Score'] = (
            ((df_signals['RSI_Signal'] * 0.35) + 
             (df_signals['MACD_Signal_Val'] * 0.35) + 
             (df_signals['Trend_Signal'] * 0.30)) * w_tech_norm +
            (df_signals['Sentiment_Signal'] * w_comp_norm) +
            (df_signals['Industry_Signal'] * w_ind_norm)
        )
        df_signals['Signal_Action'] = 'HOLD'
        df_signals.loc[df_signals['Agent_Score'] >= 0.25, 'Signal_Action'] = 'BUY'
        df_signals.loc[df_signals['Agent_Score'] <= -0.25, 'Signal_Action'] = 'SELL'
        
        # Slice both df_raw and df_signals to the requested period after indicators/signals generation
        df_raw = slice_df_to_period(df_raw, time_period)
        df_signals = slice_df_to_period(df_signals, time_period)
        
        # Recommendation
        latest_rec = get_latest_recommendation(df_signals, ticker, avg_sentiment)
        
        # Append industry news sentiment explanation to reasons
        if avg_industry_sentiment > 0.15:
            latest_rec['reasons'].append(f"Industry news trends are bullish (VADER: {avg_industry_sentiment:+.2f}) on: '{industry_query}'.")
        elif avg_industry_sentiment < -0.15:
            latest_rec['reasons'].append(f"Industry news trends are bearish (VADER: {avg_industry_sentiment:+.2f}) on: '{industry_query}'.")
        else:
            latest_rec['reasons'].append(f"Industry news trends are neutral (VADER: {avg_industry_sentiment:+.2f}) on: '{industry_query}'.")
            
        # Add math calculation breakdown to reasons
        tech_contrib = df_signals['RSI_Signal'].iloc[-1]*0.35 + df_signals['MACD_Signal_Val'].iloc[-1]*0.35 + df_signals['Trend_Signal'].iloc[-1]*0.30
        latest_rec['reasons'].append(
            f"Mathematical Decision: Technical ({w_tech_norm:.0%}) × {tech_contrib:+.2f} + "
            f"Company News ({w_comp_norm:.0%}) × {df_signals['Sentiment_Signal'].iloc[-1]:+.1f} + "
            f"Industry News ({w_ind_norm:.0%}) × {df_signals['Industry_Signal'].iloc[-1]:+.1f} = "
            f"Final Score: **{latest_rec['score']:+.2f}**"
        )
        
        # Backtest
        backtest_results = run_backtest(df_signals, initial_capital=starting_capital)
        
        # Extract latest values for display in Academy Tab
        latest_row = df_signals.iloc[-1]
        latest_rsi = latest_row.get('RSI', 50.0)
        latest_macd_hist = latest_row.get('MACD_Hist', 0.0)
        latest_close = latest_row.get('Close', 0.0)
        latest_sma20 = latest_row.get('SMA_20', 0.0)
        latest_sma50 = latest_row.get('SMA_50', 0.0)
        
        # Signal values
        rsi_sig = df_signals['RSI_Signal'].iloc[-1]
        macd_sig = df_signals['MACD_Signal_Val'].iloc[-1]
        trend_sig = df_signals['Trend_Signal'].iloc[-1]
        comp_sent_sig = df_signals['Sentiment_Signal'].iloc[-1]
        ind_sent_sig = df_signals['Industry_Signal'].iloc[-1]
        overall_score = df_signals['Agent_Score'].iloc[-1]

    # ----------------- MAIN DASHBOARD -----------------
    
    # Ticker Share Class Details Card
    share_details = get_share_class_details(ticker, comp_info.get('name', ''))
    st.markdown(f"""
    <div style="background: rgba(0, 242, 254, 0.03); border: 1px solid rgba(0, 242, 254, 0.1); padding: 1.2rem; border-radius: 15px; margin-bottom: 1.5rem; border-left: 5px solid #00f2fe;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div>
                <h4 style="margin: 0; font-size: 1.15rem; font-weight: 600; color: #00f2fe;">🛡️ Ticker Share Class: {share_details['class']}</h4>
                <p style="margin: 0.3rem 0 0 0; color: #8f9cae; font-size: 0.92rem;">
                    <b>Voting Rights:</b> {share_details['voting']} | {share_details['rights_desc']}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Switch buttons if siblings exist
    if share_details.get('siblings'):
        cols_sib = st.columns(len(share_details['siblings']) + 5)
        cols_sib[0].markdown("<p style='font-size:0.85rem; color:#8f9cae; margin-top:0.35rem; font-weight:bold;'>Analyze Sibling Class:</p>", unsafe_allow_html=True)
        for idx, sib in enumerate(share_details['siblings']):
            if cols_sib[idx + 1].button(f"🔄 Switch to {sib}", key=f"hdr_sib_btn_{sib}"):
                st.session_state.ticker_input = sib
                st.rerun()
                
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = latest_rec.get('price', 0.0)
        prev_close = df_raw['Close'].iloc[-2] if len(df_raw) > 1 else current_price
        if pd.isna(prev_close) or prev_close == 0:
            non_null_closes = df_raw['Close'].dropna()
            prev_close = non_null_closes.iloc[-2] if len(non_null_closes) > 1 else current_price
            
        change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
        st.metric(
            label="Current Stock Price",
            value=f"${current_price:,.2f}",
            delta=f"{change_pct:+.2f}% (Daily)"
        )
        
    with col2:
        action = latest_rec.get('action', 'HOLD')
        score = latest_rec.get('score', 0.0)
        
        if action == 'BUY':
            action_badge = f'<span class="badge badge-buy">BUY (Score: {score:+.2f})</span>'
        elif action == 'SELL':
            action_badge = f'<span class="badge badge-sell">SELL (Score: {score:+.2f})</span>'
        else:
            action_badge = f'<span class="badge badge-hold">HOLD (Score: {score:+.2f})</span>'
            
        st.markdown("<p style='font-size:0.9rem; color:#8f9cae; margin-bottom:0.4rem;'>Agent Decision Recommendation</p>", unsafe_allow_html=True)
        st.markdown(action_badge, unsafe_allow_html=True)
        
    with col3:
        st.metric(
            label="Company Sentiment Score",
            value=f"{avg_sentiment:+.2f}",
            delta="Bullish" if avg_sentiment > 0.15 else ("Bearish" if avg_sentiment < -0.15 else "Neutral")
        )
        
    with col4:
        strat_return = backtest_results.get('strategy_return_pct', 0.0)
        bh_return = backtest_results.get('benchmark_return_pct', 0.0)
        delta_return = strat_return - bh_return
        st.metric(
            label="Backtest Returns",
            value=f"{strat_return:.1f}%",
            delta=f"{delta_return:+.1f}% vs Buy&Hold"
        )
        
    # Prepare statistics for AI widgets
    stock_stats = {
        'name': comp_info.get('name', ticker),
        'price': latest_rec.get('price', 0.0),
        'change_pct': change_pct,
        'rsi': latest_rsi,
        'macd_hist': latest_macd_hist,
        'sentiment': avg_sentiment,
        'news_summary': "\n".join([f"- {item['title']} (by {item['publisher']})" for item in news_items[:5]]) if news_items else "No news available"
    }

    # Main Tabs
    tab_charts, tab_backtest, tab_sentiment, tab_compare, tab_academy, tab_search, tab_multiagent = st.tabs([
        "📊 Technical Charts", 
        "📈 Performance Backtester", 
        "📰 News & Sentiment",
        "⚔️ Stock Comparison",
        "🎓 Stock Analysis Academy",
        "🌐 Global Ticker Search",
        "🤖 Multi-Agent Consensus Hub"
    ])
    
    # --- TECHNICAL CHARTS TAB ---
    with tab_charts:
        st.markdown("### Stock Price & Technical Indicators")
        
        # Subplots structure: Top (Price + Bollinger Bands + Signals), Bottom (RSI & MACD)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.55, 0.22, 0.22])
        
        # Main Candlestick / Close Price
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['Close'], name='Close Price', line=dict(color='#4facfe', width=2)), row=1, col=1)
        
        # SMA Lines
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['SMA_20'], name='SMA 20', line=dict(color='#ffd200', width=1.2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['SMA_50'], name='SMA 50', line=dict(color='#ff4b2b', width=1.2, dash='dot')), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['BB_Upper'], name='BB Upper', line=dict(color='rgba(255, 255, 255, 0.15)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['BB_Lower'], name='BB Lower', line=dict(color='rgba(255, 255, 255, 0.15)', width=1), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.02)'), row=1, col=1)
        
        # BUY Signals (Markers)
        buys = df_signals[df_signals['Signal_Action'] == 'BUY']
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Close'], mode='markers', name='BUY Signal',
            marker=dict(symbol='triangle-up', size=12, color='#00b09b', line=dict(width=1, color='white'))
        ), row=1, col=1)
        
        # SELL Signals (Markers)
        sells = df_signals[df_signals['Signal_Action'] == 'SELL']
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['Close'], mode='markers', name='SELL Signal',
            marker=dict(symbol='triangle-down', size=12, color='#ff416c', line=dict(width=1, color='white'))
        ), row=1, col=1)
        
        # RSI Plot
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['RSI'], name='RSI', line=dict(color='#a18cd1', width=1.5)), row=2, col=1)
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="#ff416c", row=2, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="#00b09b", row=2, col=1)
        fig.update_yaxes(range=[10, 90], row=2, col=1)
        
        # MACD Plot
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['MACD'], name='MACD', line=dict(color='#00f2fe', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['MACD_Signal'], name='Signal Line', line=dict(color='#f7971e', width=1)), row=3, col=1)
        fig.add_trace(go.Bar(x=df_signals.index, y=df_signals['MACD_Hist'], name='MACD Histogram', marker_color='rgba(255, 255, 255, 0.2)'), row=3, col=1)
        
        # Styling Layout
        fig.update_layout(
            template="plotly_dark",
            height=700,
            margin=dict(l=20, r=20, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Elegant collapsible guide for chart symbols & indicators
        with st.expander("ℹ️ Learn to Read this Chart: Symbol & Indicator Legend", expanded=False):
            st.markdown("""
            <p style="font-size:0.95rem; color:#8f9cae; margin-bottom:1.2rem;">
                Select a tab below to inspect the mathematical formulas, definitions, and actionable strategies for each indicator plotted on the charts.
            </p>
            """, unsafe_allow_html=True)
            
            tab_leg_signals, tab_leg_ma, tab_leg_vol, tab_leg_mom = st.tabs([
                "📍 Trading Signals",
                "📈 Moving Averages",
                "🎯 Bollinger Bands",
                "📊 RSI & MACD"
            ])
            
            with tab_leg_signals:
                st.markdown(r"""
                ### 📍 Trading Signals & Price Action
                
                * <span style="background: rgba(0, 176, 155, 0.15); border: 1px solid #00b09b; color: #00b09b; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; min-width: 50px; text-align: center; display: inline-block;">▲ BUY</span> **Signal**
                  * **Meaning & Math:** Plotted when the multi-agent decision model score ($Score_{Agent} \ge +0.25$). The score is a weighted consensus of technical metrics, company-specific sentiment, and macroeconomic/industry trends.
                  * **How to Use & Benefit:** Represents a highly favorable entry zone. Investors can utilize these markers as potential buy entry signals, indicating a strong alignment of bullish indicators.
                * <span style="background: rgba(255, 65, 108, 0.15); border: 1px solid #ff416c; color: #ff416c; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; min-width: 50px; text-align: center; display: inline-block;">▼ SELL</span> **Signal**
                  * **Meaning & Math:** Plotted when the multi-agent decision model score ($Score_{Agent} \le -0.25$).
                  * **How to Use & Benefit:** Represents a warning or potential profit-taking zone. Investors can use this to secure profits, set tighter stop-losses, or explore short positions.
                * <span style="border-bottom: 2px solid #4facfe; width: 30px; display: inline-block; vertical-align: middle; margin-right: 8px;"></span> **Close Price**
                  * **Meaning & Math:** The final transaction price recorded for the stock at the end of the trading day.
                  * **How to Use & Benefit:** Used as the baseline pricing reference. Looking at close price support (historical price floors where buying stops declines) and resistance (price ceilings where selling stops rallies) helps determine key breakouts.
                """, unsafe_allow_html=True)
                
            with tab_leg_ma:
                st.markdown(r"""
                ### 📈 Moving Averages (Trend Lines)
                
                * <span style="border-bottom: 2px dashed #ffd200; width: 30px; display: inline-block; vertical-align: middle; margin-right: 8px;"></span> **SMA 20 (Simple Moving Average)**
                  * **Meaning & Math:** The 20-day rolling average of closing prices. 
                    $$SMA_{20} = \frac{1}{20}\sum_{i=1}^{20} Close_{t-i+1}$$
                  * **How to Use & Benefit:** Identifies the short-term trend direction. If the stock price stays above the SMA 20, the short-term momentum is bullish.
                * <span style="border-bottom: 2px dotted #ff4b2b; width: 30px; display: inline-block; vertical-align: middle; margin-right: 8px;"></span> **SMA 50 (Simple Moving Average)**
                  * **Meaning & Math:** The 50-day rolling average of closing prices. 
                    $$SMA_{50} = \frac{1}{50}\sum_{i=1}^{50} Close_{t-i+1}$$
                  * **How to Use & Benefit:** Identifies the medium-term trend direction.
                * **💡 Actionable Crossover Strategy:**
                  * **Golden Cross (Bullish):** When the fast-moving SMA 20 crosses *above* the slow-moving SMA 50. This indicates upward momentum is accelerating—typically a strong buy signal.
                  * **Death Cross (Bearish):** When the SMA 20 crosses *below* the SMA 50. This indicates downward momentum is accelerating—typically a signal to sell or reduce exposure.
                """, unsafe_allow_html=True)
                
            with tab_leg_vol:
                st.markdown(r"""
                ### 🎯 Bollinger Bands (Volatility & Envelopes)
                
                * <span style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.15); width: 30px; height: 16px; display: inline-block; border-radius: 2px; vertical-align: middle; margin-right: 8px;"></span> **Bollinger Bands Channel**
                  * **Meaning & Math:** A volatility envelope plotted at standard deviations above and below the SMA 20.
                    $$\text{Middle Band} = SMA_{20}$$
                    $$\text{Upper Band} = SMA_{20} + (2 \times \sigma_{20})$$
                    $$\text{Lower Band} = SMA_{20} - (2 \times \sigma_{20})$$
                    *(where \sigma_{20} is the 20-day standard deviation of the stock price)*
                  * **How to Use & Benefit:**
                    * **Volatility Squeeze:** When upper and lower bands contract tightly together (narrow gap), it indicates low volatility. This often acts as a precursor to a major price breakout. Preparing a trade for this breakout can yield large moves.
                    * **Overbought/Oversold Reversals:** Statistically, about 95% of price action remains within the bands. If the price touches the Upper Band, the asset is relatively expensive/overbought (consider taking profit). If it touches the Lower Band, it is relatively cheap/oversold (consider entering a position).
                """, unsafe_allow_html=True)
                
            with tab_leg_mom:
                st.markdown(r"""
                ### 📊 Momentum & Trend Oscillators
                
                * <span style="border-bottom: 2px solid #a18cd1; width: 30px; display: inline-block; vertical-align: middle; margin-right: 8px;"></span> **RSI (Relative Strength Index)**
                  * **Meaning & Math:** A momentum oscillator that scales price changes from 0 to 100.
                    $$RSI = 100 - \left(\frac{100}{1 + RS}\right)$$
                    $$\text{where } RS = \frac{\text{Average Gain of Up Days}}{\text{Average Loss of Down Days}} \text{ over 14 periods.}$$
                  * **How to Use & Benefit:**
                    * **Oversold (< 30):** Indicates the stock is heavily sold off and due for a bullish correction. **Benefit:** Look to purchase shares at a discount.
                    * **Overbought (> 70):** Indicates excessive buying momentum that might be exhausted. **Benefit:** Consider lock-in profit before a pullback occurs.
                * <span style="border-bottom: 2px solid #00f2fe; width: 30px; display: inline-block; vertical-align: middle; margin-right: 8px;"></span> **MACD Line &** <span style="border-bottom: 2px solid #f7971e; width: 30px; display: inline-block; vertical-align: middle; margin-right: 8px;"></span> **Signal Line**
                  * **Meaning & Math:** 
                    $$\text{MACD Line} = EMA_{12}(Close) - EMA_{26}(Close)$$
                    $$\text{Signal Line} = EMA_{9}(\text{MACD Line})$$
                    $$\text{Histogram (Gray Bars)} = \text{MACD Line} - \text{Signal Line}$$
                  * **How to Use & Benefit:**
                    * **Bullish Crossover:** When the cyan MACD line crosses *above* the orange Signal Line, the gray histogram turns positive. This shows buyers are gaining control. **Benefit:** Open long positions.
                    * **Bearish Crossover:** When the MACD line crosses *below* the Signal Line, the histogram turns negative, indicating sellers are gaining control. **Benefit:** Exit or reduce holdings.
                """, unsafe_allow_html=True)
        
        # Display Agent Recommendation Reasoning Card
        st.markdown("### Agent Signal Rationale")
        rec_col1, rec_col2 = st.columns([1, 2])
        with rec_col1:
            st.info(f"**Ticker Analyzed:** {ticker}\n\n**Signal Generated:** {action}\n\n**Date of Analysis:** {latest_rec.get('timestamp')}")
        with rec_col2:
            st.success("**Reasoning Log:**")
            for reason in latest_rec.get('reasons', []):
                st.markdown(f"- {reason}")
                
        # Render AI Assistant Widget
        render_ai_assistant_widget(
            tab_id="charts",
            tab_context=f"technical price charts, candlestick patterns, moving averages (SMA 20/50), Bollinger Bands, RSI momentum, and MACD trend crossovers for {ticker}.",
            ticker=ticker,
            stock_stats=stock_stats,
            api_key=gemini_api_key,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought
        )
                
    # --- PERFORMANCE BACKTESTER TAB ---
    with tab_backtest:
        st.markdown("### Strategy Backtest Simulation")
        
        col_res1, col_res2 = st.columns([3, 1])
        
        with col_res1:
            # Portfolio comparison chart
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=backtest_results['df_results'].index, y=backtest_results['df_results']['Portfolio_Value'], name='Agent Strategy Value', line=dict(color='#00b09b', width=2.5)))
            fig_perf.add_trace(go.Scatter(x=backtest_results['df_results'].index, y=backtest_results['df_results']['Buy_Hold_Value'], name='Benchmark Buy & Hold', line=dict(color='#8f9cae', width=1.5, dash='dash')))
            
            fig_perf.update_layout(
                template="plotly_dark",
                height=450,
                margin=dict(l=20, r=20, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            fig_perf.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
            fig_perf.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with col_res2:
            # Strategy Metrics Breakdown
            st.markdown("<p style='font-size:1.2rem; font-weight:600; color:#00f2fe;'>Risk / Reward Profile</p>", unsafe_allow_html=True)
            st.markdown(f"**Starting Capital:** `${starting_capital:,.2f}`")
            st.markdown(f"**Ending Capital:** `${backtest_results['final_value']:,.2f}`")
            st.markdown(f"**Strategy Return:** `{backtest_results['strategy_return_pct']:.2f}%`")
            st.markdown(f"**Benchmark Return:** `{backtest_results['benchmark_return_pct']:.2f}%`")
            st.markdown(f"**Annual Sharpe Ratio:** `{backtest_results['sharpe_ratio']:.2f}`")
            st.markdown(f"**Maximum Drawdown:** `{backtest_results['max_drawdown_pct']:.2f}%`")
            st.markdown(f"**Trades Triggered:** `{backtest_results['total_trades']}`")
            
        # Display list of executed trades
        st.markdown("### Agent Transaction History")
        trades_list = backtest_results.get('trades', [])
        if trades_list:
            df_trades = pd.DataFrame(trades_list)
            # Standardize columns for display
            df_trades.columns = ['Date', 'Transaction Type', 'Price per Share ($)', 'Portfolio Capital ($)']
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.write("No trades executed during this time frame. The agent held cash or stayed in a single state.")
            
        # Render AI Assistant Widget
        render_ai_assistant_widget(
            tab_id="backtest",
            tab_context=f"historical strategy backtesting results, including starting capital (${starting_capital}), ending strategy portfolio value (${backtest_results['final_value']}), benchmark returns (Buy & Hold), maximum drawdown, Sharpe ratio, win rate, and transaction history logs for {ticker}.",
            ticker=ticker,
            stock_stats=stock_stats,
            api_key=gemini_api_key,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought
        )

    # --- NEWS & SENTIMENT TAB ---
    with tab_sentiment:
        st.markdown("### Live Financial News Headlines & Sentiment Parsing")
        
        # Helper function to process and sort news by risk
        def process_and_sort_news(news_list):
            processed = []
            for item in news_list:
                title = item.get('title', '')
                scores = analyze_headline_sentiment(title)
                comp = scores['compound']
                
                # High Risk: Sentiment < -0.1 (Red)
                # Medium Risk: Sentiment between -0.1 and 0.2 (Yellow)
                # Low Risk: Sentiment >= 0.2 (Green)
                if comp < -0.1:
                    risk_level = "High Risk"
                    risk_color = "#ff416c"  # Red
                    risk_sort = 0
                elif comp >= 0.2:
                    risk_level = "Low Risk"
                    risk_color = "#00b09b"  # Green
                    risk_sort = 2
                else:
                    risk_level = "Medium Risk"
                    risk_color = "#ffd200"  # Yellow
                    risk_sort = 1
                    
                processed.append({
                    'item': item,
                    'title': title,
                    'score': comp,
                    'publisher': item.get('publisher', 'Unknown'),
                    'link': item.get('link', '#'),
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'risk_sort': risk_sort
                })
            # Sort: High Risk first (0), then Medium (1), then Low (2)
            processed.sort(key=lambda x: x['risk_sort'])
            return processed

        sorted_company_news = process_and_sort_news(news_items)
        sorted_industry_news = process_and_sort_news(ind_news_items)

        # News Headlines side-by-side
        col_news1, col_news2 = st.columns([1, 1])
        
        with col_news1:
            st.markdown(f"<h4 style='font-weight:600;'>🏢 {ticker} News Headlines (High to Low Risk)</h4>", unsafe_allow_html=True)
            if not sorted_company_news:
                st.write("No company-specific news items retrieved from Yahoo Finance.")
            else:
                for processed_item in sorted_company_news[:8]:
                    title = processed_item['title']
                    comp = processed_item['score']
                    risk_level = processed_item['risk_level']
                    risk_color = processed_item['risk_color']
                    publisher = processed_item['publisher']
                    link = processed_item['link']
                    
                    st.markdown(f"""
                    <div class="news-card" style="border-left: 4px solid {risk_color};">
                        <p style="margin:0 0 0.3rem 0; font-weight:600; font-size:0.95rem;">
                            <a href="{link}" target="_blank" style="text-decoration:none; color:#e1e7f0;">{title}</a>
                        </p>
                        <p style="margin:0; font-size:0.8rem; color:#8f9cae;">
                            Publisher: {publisher} | Score: {comp:+.2f} | 
                            <span style="color:{risk_color}; font-weight:700;">{risk_level}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with col_news2:
            st.markdown(f"<h4 style='font-weight:600;'>🌐 Industry News: '{industry_query}' (High to Low Risk)</h4>", unsafe_allow_html=True)
            if not sorted_industry_news:
                st.write("No industry-specific news headlines retrieved from Google News.")
            else:
                for processed_item in sorted_industry_news[:8]:
                    title = processed_item['title']
                    comp = processed_item['score']
                    risk_level = processed_item['risk_level']
                    risk_color = processed_item['risk_color']
                    publisher = processed_item['publisher']
                    link = processed_item['link']
                    
                    st.markdown(f"""
                    <div class="news-card" style="border-left: 4px solid {risk_color};">
                        <p style="margin:0 0 0.3rem 0; font-weight:600; font-size:0.95rem;">
                            <a href="{link}" target="_blank" style="text-decoration:none; color:#e1e7f0;">{title}</a>
                        </p>
                        <p style="margin:0; font-size:0.8rem; color:#8f9cae;">
                            Publisher: {publisher} | Score: {comp:+.2f} | 
                            <span style="color:{risk_color}; font-weight:700;">{risk_level}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Aggregated Sentiment Distribution Comparison")
        
        col_hist1, col_hist2 = st.columns([1, 1])
        
        with col_hist1:
            st.markdown(f"<h5 style='font-weight:600; text-align:center;'>🏢 {ticker} Sentiment (Avg: {avg_sentiment:+.2f})</h5>", unsafe_allow_html=True)
            if news_items:
                comp_scores = [analyze_headline_sentiment(item.get('title', ''))['compound'] for item in news_items]
                fig_comp_hist = go.Figure()
                fig_comp_hist.add_trace(go.Histogram(
                    x=comp_scores, nbinsx=10, marker=dict(color='#4facfe', line=dict(width=1, color='white')), opacity=0.8
                ))
                fig_comp_hist.update_layout(
                    template="plotly_dark", height=250, margin=dict(l=20, r=20, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="VADER Sentiment Score", yaxis_title="Count"
                )
                st.plotly_chart(fig_comp_hist, use_container_width=True)
                
        with col_hist2:
            st.markdown(f"<h5 style='font-weight:600; text-align:center;'>🌐 Industry Sentiment (Avg: {avg_industry_sentiment:+.2f})</h5>", unsafe_allow_html=True)
            if ind_news_items:
                ind_scores = [analyze_headline_sentiment(item.get('title', ''))['compound'] for item in ind_news_items]
                fig_ind_hist = go.Figure()
                fig_ind_hist.add_trace(go.Histogram(
                    x=ind_scores, nbinsx=10, marker=dict(color='#a18cd1', line=dict(width=1, color='white')), opacity=0.8
                ))
                fig_ind_hist.update_layout(
                    template="plotly_dark", height=250, margin=dict(l=20, r=20, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="VADER Sentiment Score", yaxis_title="Count"
                )
                st.plotly_chart(fig_ind_hist, use_container_width=True)
                
        st.markdown("""
        **VADER Sentiment Interpretation:**
        - **+0.05 to +1.0**: Positive Sentiment (Bullish)
        - **-0.05 to +0.05**: Neutral Sentiment
        - **-1.0 to -0.05**: Negative Sentiment (Bearish)
        """)
        
        # Render AI Assistant Widget
        render_ai_assistant_widget(
            tab_id="sentiment",
            tab_context=f"financial news and market sentiment analysis, including VADER sentiment scores (Compound, Positive, Negative, Neutral) for company news headlines and sector/industry news trends on: '{industry_query}'.",
            ticker=ticker,
            stock_stats=stock_stats,
            api_key=gemini_api_key,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought
        )

    # --- STOCK COMPARISON DUEL TAB ---
    with tab_compare:
        st.markdown("## ⚔️ Stock Comparison Duel")
        st.markdown("""
        Compare any multiple global stock symbols side-by-side to judge which one has stronger momentum, better news sentiment, lower volatility, and higher AI Agent recommendation scores.
        """)
        
        # User input for comparison tickers
        # Default is: current ticker and a sibling or TSLA
        default_compare_value = f"{ticker}, TSLA" if ticker != "TSLA" else "TSLA, AAPL"
        
        compare_input = st.text_input(
            "🏢 Enter Tickers to Compare (comma-separated, e.g., AAPL, TSLA, MSFT, GOOG, GOOGL)",
            value=default_compare_value,
            key="compare_tickers_input_text"
        )
        
        if compare_input:
            input_symbols = [s.strip().upper() for s in compare_input.split(",") if s.strip()]
            
            if len(input_symbols) < 2:
                st.warning("Please enter at least 2 tickers to run a side-by-side comparison.")
            else:
                with st.spinner("Analyzing and comparing tickers..."):
                    comparison_results = []
                    valid_symbols = []
                    
                    for sym in input_symbols:
                        res = run_agent_analysis_for_comparison(sym, period=time_period)
                        if 'error' in res:
                            st.error(f"❌ {res['error']}")
                        else:
                            comparison_results.append(res)
                            valid_symbols.append(res['ticker'])
                    
                    if len(comparison_results) >= 2:
                        # Display Comparative Data Cards / Table
                        st.markdown("### 📊 Metrics Matrix")
                        
                        # Build a clean dataframe for comparison
                        compare_rows = []
                        for res in comparison_results:
                            action_disp = f"🟢 BUY ({res['score']:+.2f})" if res['action'] == 'BUY' else (f"🔴 SELL ({res['score']:+.2f})" if res['action'] == 'SELL' else f"🟡 HOLD ({res['score']:+.2f})")
                            
                            compare_rows.append({
                                'Ticker': res['ticker'],
                                'Company Name': res['name'],
                                'Share Class / Exchange': f"{res['share_class']} ({res['sector']})",
                                'Current Price': f"${res['price']:,.2f}",
                                'Daily Change': f"{res['change_pct']:+.2f}%",
                                f'{time_period} Return': f"{res['period_return']:+.2f}%",
                                'Volatility (Std Dev)': f"{res['volatility']:.2f}%",
                                'Max Drawdown': f"{res['max_drawdown']:+.2f}%",
                                'RSI (14)': f"{res['rsi']:.1f}",
                                'MACD Trend': "Bullish Crossover" if res['macd_hist'] > 0 else "Bearish Trend",
                                'Sentiment Score': f"{res['sentiment']:+.2f}",
                                'Agent Recommendation': action_disp
                            })
                            
                        df_compare_tbl = pd.DataFrame(compare_rows)
                        st.dataframe(df_compare_tbl, use_container_width=True, hide_index=True)
                        
                        # Winner Declaration
                        # Highest score wins
                        sorted_res = sorted(comparison_results, key=lambda x: x['score'], reverse=True)
                        winner = sorted_res[0]
                        runner_up = sorted_res[1]
                        
                        st.markdown(f"""
                        <div style="background: rgba(0, 176, 155, 0.05); border: 1px solid rgba(0, 176, 155, 0.2); padding: 1.5rem; border-radius: 15px; margin-top: 1.5rem; border-left: 5px solid #00b09b;">
                            <h3 style="margin: 0 0 0.5rem 0; color: #00b09b;">🏆 Agent Performance Recommendation: <b>{winner['name']} ({winner['ticker']})</b></h3>
                            <p style="margin: 0; color: #e1e7f0; font-size: 1.05rem;">
                                Based on a mathematical synthesis of technical indicators, news sentiment, and historical performance, <b>{winner['ticker']}</b> is the highest ranked stock with a decision score of <b>{winner['score']:+.2f}</b> (Action: <b>{winner['action']}</b>).
                            </p>
                            <p style="margin: 0.5rem 0 0 0; color: #8f9cae; font-size: 0.95rem;">
                                Runner up: <b>{runner_up['name']} ({runner_up['ticker']})</b> with a decision score of <b>{runner_up['score']:+.2f}</b> (Action: <b>{runner_up['action']}</b>).
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Graph relative performance
                        st.markdown("### 📈 Relative Price Performance Chart")
                        fig_compare_rel = go.Figure()
                        
                        for res in comparison_results:
                            hist_df = res['history']
                            if not hist_df.empty:
                                norm_close = (hist_df['Close'] / hist_df['Close'].iloc[0]) * 100
                                fig_compare_rel.add_trace(go.Scatter(
                                    x=hist_df.index, y=norm_close, name=f"{res['ticker']} ({res['name']})",
                                    mode='lines', line=dict(width=2.5)
                                ))
                                
                        fig_compare_rel.update_layout(
                            template="plotly_dark",
                            height=450,
                            margin=dict(l=20, r=20, t=10, b=10),
                            xaxis_title="Date",
                            yaxis_title="Relative Return % (Base 100)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        fig_compare_rel.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
                        fig_compare_rel.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
                        
                        st.plotly_chart(fig_compare_rel, use_container_width=True)
                        
                        # Share class explanation if compared symbols are class differences
                        diff_expl = get_ticker_difference_explanation(valid_symbols)
                        if diff_expl:
                            st.info(diff_expl)
                            
                        # Render AI Assistant Widget for comparison tab
                        compare_names = ", ".join([res['name'] for res in comparison_results])
                        compare_stats = {
                            'name': compare_names,
                            'price': winner['price'],
                            'change_pct': winner['change_pct'],
                            'rsi': winner['rsi'],
                            'macd_hist': winner['macd_hist'],
                            'sentiment': winner['sentiment'],
                            'news_summary': f"Comparing performance, returns, volatility, drawdowns, and agent recommendations between: {', '.join(valid_symbols)}."
                        }
                        render_ai_assistant_widget(
                            tab_id="compare",
                            tab_context=f"cross-exchange and multi-stock comparison metrics, comparing returns, volatility, drawdowns, and technical/sentiment buy/sell signals between multiple symbols: {', '.join(valid_symbols)}.",
                            ticker=ticker,
                            stock_stats=compare_stats,
                            api_key=gemini_api_key,
                            rsi_oversold=rsi_oversold,
                            rsi_overbought=rsi_overbought
                        )

    # --- STOCK ANALYSIS ACADEMY TAB ---
    with tab_academy:
        st.markdown("## 🎓 Stock Analysis Academy")
        st.markdown("""
        Welcome to the educational hub! Here, you can learn the exact mathematical models, formulas, and reasoning rules the AI agent uses to analyze stocks in real-time.
        """)
        
        # 1. Live Solver Breakdown
        st.markdown("### 🧮 Live Strategy Solver")
        st.markdown(f"Below is the live, step-by-step mathematical calculation for **{ticker}** based on your current settings and weight inputs:")
        
        # Calculate individual weighted scores
        tech_subscore = (rsi_sig * 0.35) + (macd_sig * 0.35) + (trend_sig * 0.30)
        weighted_tech = tech_subscore * w_tech_norm
        weighted_comp = comp_sent_sig * w_comp_norm
        weighted_ind = ind_sent_sig * w_ind_norm
        
        # Create a premium comparison table
        st.markdown(f"""
        | Parameter / Indicator | Current Real-Time Value | Signal Value (-1 to +1) | Weighted Weight | Final Contribution |
        | :--- | :--- | :---: | :---: | :---: |
        | **RSI (14)** | {latest_rsi:.2f} ({"Oversold" if latest_rsi < rsi_oversold else ("Overbought" if latest_rsi > rsi_overbought else "Neutral")}) | `{rsi_sig:+.1f}` | {w_tech_norm*35:.1f}% | `{w_tech_norm*0.35*rsi_sig:+.2f}` |
        | **MACD Crossover** | Hist: {latest_macd_hist:.4f} | `{macd_sig:+.1f}` | {w_tech_norm*35:.1f}% | `{w_tech_norm*0.35*macd_sig:+.2f}` |
        | **SMA Trend** | Price: ${latest_close:.2f} (20-day SMA: ${latest_sma20:.2f}) | `{trend_sig:+.1f}` | {w_tech_norm*30:.1f}% | `{w_tech_norm*0.30*trend_sig:+.2f}` |
        | **Company News Sentiment** | VADER Compound: {avg_sentiment:+.2f} | `{comp_sent_sig:+.1f}` | {w_comp_norm*100:.1f}% | `{weighted_comp:+.2f}` |
        | **Industry / Macro Sentiment** | VADER Compound: {avg_industry_sentiment:+.2f} | `{ind_sent_sig:+.1f}` | {w_ind_norm*100:.1f}% | `{weighted_ind:+.2f}` |
        | **TOTAL AGENT SCORE** | | | **100%** | **`{overall_score:+.2f}`** |
        """)
        
        # Display Decision
        if action == 'BUY':
            action_html = '<span style="color:#00b09b; font-weight:bold;">BUY 🟢</span>'
            explanation = f"Since the total score (**{overall_score:+.2f}**) is **>= +0.25**, the mathematical model triggers a **BUY** recommendation."
        elif action == 'SELL':
            action_html = '<span style="color:#ff416c; font-weight:bold;">SELL 🔴</span>'
            explanation = f"Since the total score (**{overall_score:+.2f}**) is **<= -0.25**, the mathematical model triggers a **SELL** recommendation."
        else:
            action_html = '<span style="color:#f7971e; font-weight:bold;">HOLD 🟡</span>'
            explanation = f"Since the total score (**{overall_score:+.2f}**) lies between **-0.25 and +0.25**, the model recommends to **HOLD** cash/position."
            
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 15px; margin-top: 1rem;">
            <p style="margin:0; font-size:1.1rem;">
                <b>Mathematical Decision:</b> {action_html} (Final Score: <b>{overall_score:+.2f}</b>)
            </p>
            <p style="margin: 0.5rem 0 0 0; color:#8f9cae; font-size:0.95rem;">
                {explanation}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Educational content: How indicators work
        st.markdown("### 📊 How These Indicators Work Under the Hood")
        
        col_ed1, col_ed2 = st.columns(2)
        
        with col_ed1:
            with st.expander("📈 RSI (Relative Strength Index)"):
                st.markdown("""
                **What is it?**
                The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.
                
                **The Math:**
                $$RSI = 100 - \\frac{100}{1 + RS}$$
                Where:
                $$RS = \\frac{\\text{Average Gain of Up Days}}{\\text{Average Loss of Down Days}}$$
                
                **How we analyze it:**
                * **Oversold (< 30)**: Indicates the asset might be undervalued and due for a bullish correction (we trigger a `+1.0` signal).
                * **Overbought (> 70)**: Indicates the asset might be overvalued and due for a bearish pullback (we trigger a `-1.0` signal).
                * **Neutral (30 to 70)**: Trend is stable (we trigger a `0.0` signal).
                """)
                
            with st.expander("📉 MACD (Moving Average Convergence Divergence)"):
                st.markdown("""
                **What is it?**
                MACD is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a stock's price.
                
                **The Math:**
                1. **MACD Line**: $EMA_{12}(Close) - EMA_{26}(Close)$
                2. **Signal Line**: $EMA_{9}(MACD)$
                3. **MACD Histogram**: $MACD - Signal$
                
                **How we analyze it:**
                * **Bullish Crossover**: When the MACD Line crosses *above* the Signal Line (Histogram goes from negative to positive). This signals upward momentum (`+1.0` signal).
                * **Bearish Crossover**: When the MACD Line crosses *below* the Signal Line (Histogram goes from positive to negative). This signals downward momentum (`-1.0` signal).
                """)
                
        with col_ed2:
            with st.expander("🎯 Bollinger Bands"):
                st.markdown("""
                **What is it?**
                Bollinger Bands consist of a middle band (Simple Moving Average) and two outer volatility bands calculated as standard deviations from the middle band.
                
                **The Math:**
                * **Middle Band**: $SMA_{20}(Close)$
                * **Upper Band**: $SMA_{20}(Close) + (2 \\times \\sigma_{20})$
                * **Lower Band**: $SMA_{20}(Close) - (2 \\times \\sigma_{20})$
                *(where $\\sigma_{20}$ is the 20-day standard deviation of the closing price)*
                
                **How we analyze it:**
                * Price near/above the **Upper Band** indicates high relative price/overbought volatility.
                * Price near/below the **Lower Band** indicates low relative price/oversold volatility.
                """)
                
            with st.expander("📰 VADER Sentiment Analysis"):
                st.markdown("""
                **What is it?**
                VADER (Valence Aware Dictionary and sEntiment Reasoner) is a gold-standard lexicon and rule-based sentiment analysis tool specifically attuned to social media and financial headlines.
                
                **How it works:**
                * It maps lexical features (words like *beating estimates*, *plunging*, *growth*) to sentiment intensity scores.
                * It sums the intensities and normalizes them into a single **Compound Score** between `-1.0` (extremely negative/bearish) and `+1.0` (extremely positive/bullish).
                
                **How we analyze it:**
                * **Bullish Sentiment (> +0.15)**: Market news is optimistic (triggers `+1.0` signal).
                * **Bearish Sentiment (< -0.15)**: Market news is pessimistic (triggers `-1.0` signal).
                """)
                
        # Render AI Assistant Widget
        render_ai_assistant_widget(
            tab_id="academy",
            tab_context=f"educational explanations, mathematical formulas, and calculations for indicators like RSI, MACD, Bollinger Bands, and VADER sentiment.",
            ticker=ticker,
            stock_stats=stock_stats,
            api_key=gemini_api_key,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought
        )
                
    # --- GLOBAL TICKER SEARCH & COMPARE TAB ---
    with tab_search:
        st.markdown("## 🌐 Global Ticker Search & Multi-Exchange Comparison")
        st.markdown("""
        Search for any company name in the world to instantly find its ticker symbols, list the indices/exchanges they belong to, and compare their performance side-by-side.
        """)
        
        # Search input
        search_query = st.text_input("🏢 Search Company Name (e.g., Apple, Tata, Reliance, Berkshire Hathaway)", "", key="global_company_search")
        
        if search_query:
            quotes = cached_search_ticker(search_query)
            quotes = expand_search_results(quotes, search_query)
            if not quotes:
                st.info("No matching companies or tickers found. Try a different search term.")
            else:
                st.markdown(f"### Found {len(quotes)} listings for **'{search_query}'**:")
                
                # Checkbox selection for side-by-side comparison
                st.markdown("##### 🔍 Compare Listings")
                st.caption("Check the boxes next to the tickers you want to compare side-by-side:")
                
                selected_tickers = []
                cols = st.columns(3)
                for idx, q in enumerate(quotes[:9]): # Limit to top 9 results for layout
                    sym = q.get('symbol', '')
                    exch_disp = q.get('exchDisp', 'Unknown')
                    short_name = q.get('shortname', '')
                    
                    with cols[idx % 3]:
                        # Make checkbox key unique to prevent streamlit component key collision
                        is_checked = st.checkbox(f"{sym} ({exch_disp})", key=f"compare_check_{sym}_{idx}")
                        if is_checked:
                            selected_tickers.append(q)
                
                # If tickers are selected for comparison
                if len(selected_tickers) > 1:
                    st.markdown("### 📊 Side-by-Side Listing Comparison")
                    
                    comparison_data = []
                    for q in selected_tickers:
                         sym = q['symbol']
                         try:
                             # We can fetch latest history or info (cached)
                             tick = yf.Ticker(sym)
                             hist = tick.history(period="5d")
                             if not hist.empty:
                                 current_price = hist['Close'].iloc[-1]
                                 prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                                 daily_change = ((current_price - prev_close) / prev_close) * 100
                                 volume = hist['Volume'].iloc[-1]
                             else:
                                 current_price, daily_change, volume = 0.0, 0.0, 0.0
                         except Exception:
                             current_price, daily_change, volume = 0.0, 0.0, 0.0
                             
                         comparison_data.append({
                             'Ticker': sym,
                             'Company Name': q.get('shortname', q.get('longname', '')),
                             'Index / Exchange': f"{q.get('exchDisp', 'Unknown')} ({q.get('exchange', '')})",
                             'Current Price': f"${current_price:,.2f}" if current_price > 0 else "N/A",
                             'Daily Change': f"{daily_change:+.2f}%" if current_price > 0 else "N/A",
                             'Volume': f"{volume:,.0f}" if volume > 0 else "N/A"
                         })
                         
                    df_compare = pd.DataFrame(comparison_data)
                    st.dataframe(df_compare, use_container_width=True, hide_index=True)
                    
                    # Explaining structural differences between these listings
                    ticker_list = [item['Ticker'] for item in comparison_data]
                    diff_explanation = get_ticker_difference_explanation(ticker_list)
                    if diff_explanation:
                        st.info(diff_explanation)
                    
                    # Graph comparison
                    st.markdown("#### 📈 5-Day Relative Performance Comparison")
                    fig_rel = go.Figure()
                    for q in selected_tickers:
                        sym = q['symbol']
                        try:
                            tick = yf.Ticker(sym)
                            hist = tick.history(period="5d")
                            if not hist.empty and len(hist) > 1:
                                norm_close = (hist['Close'] / hist['Close'].iloc[0]) * 100
                                fig_rel.add_trace(go.Scatter(
                                    x=hist.index, y=norm_close, name=f"{sym} ({q.get('exchDisp')})",
                                    mode='lines+markers', line=dict(width=2)
                                ))
                        except Exception:
                            pass
                            
                    fig_rel.update_layout(
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=20, r=20, t=10, b=10),
                        xaxis_title="Date",
                        yaxis_title="Relative Performance (Base 100)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🏢 Available Listings Details")
                
                # Show full card list with Apply Ticker buttons
                for idx, q in enumerate(quotes):
                    symbol = q.get('symbol', '')
                    exchange_disp = q.get('exchDisp', 'Unknown Exchange')
                    exchange_code = q.get('exchange', '')
                    short_name = q.get('shortname', '')
                    long_name = q.get('longname', short_name)
                    sector = q.get('sector', 'Unknown Sector')
                    industry = q.get('industry', 'Unknown Industry')
                    type_disp = q.get('typeDisp', 'Equity')
                    
                    # Card
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.2rem; border-radius: 15px; margin-bottom: 0.5rem; margin-top: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.2rem; font-weight: 800; color: #00f2fe;">{symbol}</span>
                                <span style="margin-left: 10px; padding: 2px 8px; background: rgba(0, 242, 254, 0.1); border-radius: 5px; font-size: 0.8rem; color: #00f2fe;">{exchange_disp} ({exchange_code})</span>
                            </div>
                            <span style="font-size: 0.8rem; color: #8f9cae;">{type_disp}</span>
                        </div>
                        <h4 style="margin: 0.4rem 0; font-size: 1.05rem; font-weight: 600; color: #ffffff;">{long_name}</h4>
                        <p style="margin: 0; font-size: 0.85rem; color: #8f9cae;">Sector: <b>{sector}</b> | Industry: <b>{industry}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Button to apply
                    if st.button(f"📊 Apply {symbol} Ticker", key=f"btn_apply_{symbol}_{idx}"):
                        st.session_state.ticker_input = symbol
                        st.rerun()
                        
                # Render AI Assistant Widget
                render_ai_assistant_widget(
                    tab_id="search",
                    tab_context=f"global stock symbol search, ticker suffixes (.NS, .BO, .L, .DE), and corporate share classes (A, B, C).",
                    ticker=ticker,
                    stock_stats=stock_stats,
                    api_key=gemini_api_key,
                    rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought
                )

    # --- MULTI-AGENT CONSENSUS HUB TAB ---
    with tab_multiagent:
        st.markdown("### 🤖 Multi-Agent Consensus Analysis Hub")
        st.markdown("Invokes four specialized quantitative and advisory agents to form a consensus valuation, competitive analysis, and investment recommendation.")
        
        # Check session state
        ma_state_key = f"multiagent_res_{ticker}"
        
        # Trigger button
        run_ma_analysis = st.button("🤖 Run Multi-Agent Consensus Analysis Panel", key="run_ma_btn")
        
        if run_ma_analysis or ma_state_key in st.session_state:
            # We calculate this on every rerun to make sure it's available for rendering the TL;DR card
            multi_trade_results = run_multi_trade_backtest(df_signals, initial_capital=starting_capital)
            
            # Execute analysis if triggered or already exists
            if run_ma_analysis or ma_state_key not in st.session_state:
                # Trading Agent: Active Multi-Trade Simulator
                with st.spinner("Invoking Active Multi-Trade Simulator..."):
                    trades_sample = multi_trade_results.get('trades', [])[:10]
                    formatted_trades_sample = []
                    for t in trades_sample:
                        formatted_trades_sample.append(
                            f"[{t['date']}] {t['type']} at ${t['price']:.2f} (Shares: {t['shares']:.2f}, PortVal: ${t['portfolio_value']:.2f})"
                        )
                    trades_context = "\n".join(formatted_trades_sample) if formatted_trades_sample else "No trades triggered"
                    
                    prompt_a5 = f"""
You are the Active Multi-Trade Simulator. Review the active multi-trade strategy results and trade logs for {ticker}.

Active Strategy Performance:
- Initial Capital: ${starting_capital:,.2f}
- Final Strategy Value: ${multi_trade_results.get('final_value', 0.0):,.2f}
- Multi-Trade Return: {multi_trade_results.get('strategy_return_pct', 0.0):.2f}%
- Benchmark (Buy & Hold) Return: {multi_trade_results.get('benchmark_return_pct', 0.0):.2f}%
- Total Trades Triggered: {multi_trade_results.get('total_trades', 0)}

Trade Logs Sample (Daily Close Execution mimicking the market):
{trades_context}

Instructions:
1. Explain this active trading strategy (scaling in 25% on BUY signals, scaling out 50% on SELL signals) and compare its return and risk (such as capital preservation) against a simple Buy & Hold.
2. Outline specific trades from the logs to explain how executing multiple trades (instead of a single all-in/all-out position) affected the portfolio value generated.
3. Keep the tone professional, adult-appropriate, and easy to understand.
4. Structure & Formatting Requirements:
   - Output in clean, highly structured Markdown.
   - Use bold subheadings (e.g., ### Section Name) for logical sections.
   - Use bullet points with bold key metrics.
   - Separate distinct ideas with spacing or horizontal rules (`---`).
   - Start directly with the structured answer (no conversational preamble).
   - IMPORTANT: DO NOT write any math symbols or equations using LaTeX delimiters like \\[, \\], \\(, \\), or double/single dollar signs ($). Write all mathematical equations and formulas in simple, plain english text representation.
"""
                    if is_active:
                        agent5_output = call_provider_llm_api(llm_provider, selected_model, prompt_a5, gemini_api_key, ollama_url)
                    else:
                        agent5_output = f"""
- **Active Execution Return**: Generated **{multi_trade_results.get('strategy_return_pct', 0.0):.2f}%** return vs **{multi_trade_results.get('benchmark_return_pct', 0.0):.2f}%** benchmark.
- **Trades Executed**: Triggered a total of **{multi_trade_results.get('total_trades', 0)}** active market-mimicking scale-in/scale-out events.
- **Allocation Value**: Ended with a total capital of **${multi_trade_results.get('final_value', 0.0):,.2f}** from an initial **${starting_capital:,.2f}**.
- **Execution Profile**: Splits capital into 25% tranches on BUY signals to buy at different price levels, and sells 50% of the position on SELL signals to lock in partial profits.
"""

                # Chart Intelligence Analyst
                with st.spinner("Invoking Chart Intelligence Analyst..."):
                    prompt_a1 = f"""
You are the Chart Intelligence Analyst. Provide a complete and precise summary of the price action and technical indicators for {ticker} (Sector: {comp_info.get('sector', 'Unknown')}, Industry: {comp_info.get('industry', 'Unknown')}).

Active Multi-Trade Simulator Data:
---
{agent5_output}
---

Context & Latest Data:
- Candle: Date={latest_rec.get('timestamp')}, Close=${latest_row.get('Close', 0.0):,.2f}, Volume={latest_row.get('Volume', 0.0):,}
- Indicators: SMA_20=${latest_row.get('SMA_20', 0.0):,.2f}, SMA_50=${latest_row.get('SMA_50', 0.0):,.2f}
- Bollinger Bands: Upper=${latest_row.get('BB_Upper', 0.0):,.2f}, Mid=${latest_row.get('BB_Middle', 0.0):,.2f}, Lower=${latest_row.get('BB_Lower', 0.0):,.2f}
- Momentum: RSI(14)={latest_row.get('RSI', 50.0):.2f}, MACD_Hist={latest_row.get('MACD_Hist', 0.0):.4f}

Instructions:
1. Provide a complete, fully formed explanation of trend status, RSI momentum, Bollinger Band position, and MACD crossover.
2. Integrate Trading Agent Data: Analyze how the active trading simulator's specific trade entry/exit dates and execution prices align with these technical chart trendlines and support/resistance boundaries (e.g. did it buy near SMA support or sell near BB upper band resistance).
3. Explain technical patterns using simple analogies (like a runner's speed for RSI, or rubber bands stretching for Bollinger Bands) in a professional tone suitable for adults.
4. Structure & Formatting Requirements:
   - Output in clean, highly structured Markdown.
   - Use bold subheadings (e.g., ### Section Name) for logical sections.
   - Use bullet points with bold key metrics.
   - Separate distinct ideas with spacing or horizontal rules (`---`).
   - Do not output any HTML tags.
   - Start directly with the structured answer (no conversational preamble).
   - IMPORTANT: DO NOT write any math symbols or equations using LaTeX delimiters like \\[, \\], \\(, \\), or double/single dollar signs ($). Write all mathematical equations and formulas in simple, plain english text representation.
"""
                    if is_active:
                        agent1_output = call_provider_llm_api(llm_provider, selected_model, prompt_a1, gemini_api_key, ollama_url)
                    else:
                        # Local expert fallback report
                        trend_desc = "UPTREND" if latest_row.get('Close', 0.0) > latest_row.get('SMA_20', 0.0) else "DOWNTREND"
                        rsi_desc = "OVERSOLD" if latest_rsi < rsi_oversold else ("OVERBOUGHT" if latest_rsi > rsi_overbought else "NEUTRAL")
                        macd_desc = "BULLISH Crossover" if latest_macd_hist > 0 else "BEARISH Crossover"
                        bb_desc = "trading near Upper Bollinger Band" if latest_close > latest_row.get('BB_Middle', 0.0) else "trading near Lower Bollinger Band"
                        
                        agent1_output = f"""
- **Trend Analysis**: The stock price is currently in a **{trend_desc}**, trading relative to the 20-day SMA boundary.
- **RSI Momentum**: The Relative Strength Index (RSI) is at **{latest_rsi:.2f}**, which indicates **{rsi_desc}** status.
- **MACD Trend**: A **{macd_desc}** is observed, with the histogram at **{latest_macd_hist:.4f}**.
- **Volatility Boundaries**: The price is **{bb_desc}**, indicating bounded range volatility movement.
- **Trading Simulator Alignment**: The Active Multi-Trade Simulator's scale-in/scale-out decisions corresponded to these trend conditions, buying tranches during support tests and selling during momentum reversals.
"""
                
                # Quantitative Metrics Parser
                with st.spinner("Invoking Quantitative Metrics Parser..."):
                    prompt_a2 = f"""
You are the Quantitative Metrics Parser. Perform a precise mathematical review of backtest execution and sentiment statistics for {ticker}.

Active Multi-Trade Simulator Data:
---
{agent5_output}
---

Metrics Context:
- Capital: Starting=${starting_capital:,.2f}, Strategy Final=${backtest_results.get('final_value', 0.0):,.2f}
- Return: Strategy={backtest_results.get('strategy_return_pct', 0.0):.2f}%, Benchmark (Buy & Hold)={backtest_results.get('benchmark_return_pct', 0.0):.2f}%, Excess={backtest_results.get('strategy_return_pct', 0.0) - backtest_results.get('benchmark_return_pct', 0.0):+.2f}%
- Risk: Max Drawdown={backtest_results.get('max_drawdown_pct', 0.0):.2f}%, Volatility={backtest_results.get('volatility_pct', 0.0):.2f}%, Trades={backtest_results.get('total_trades', 0)}
- Sentiment: Company Sentiment={avg_sentiment:+.2f}, Industry Sentiment={avg_industry_sentiment:+.2f}, Combined Score={latest_rec.get('score', 0.0):+.2f}

Instructions:
1. Provide a complete, fully formed quantitative analysis of returns, risks (drawdown and volatility), and sentiment scores.
2. Integrate Trading Agent Data: Compare the risk/reward and efficiency of the active multi-trade simulator vs simple Buy & Hold and all-in/all-out systems, showing mathematically how the multiple executions altered drawdowns and Sharpe Ratio.
3. Translate mathematical details into simple intuitive concepts (like savings account comparisons or risk protections) in mature, professional language.
4. Structure & Formatting Requirements:
   - Output in clean, highly structured Markdown.
   - Use bold subheadings (e.g., ### Section Name) for logical sections.
   - Use bullet points with bold key metrics.
   - Separate distinct ideas with spacing or horizontal rules (`---`).
   - Do not output any HTML tags.
   - Start directly with the structured answer (no conversational preamble).
   - IMPORTANT: DO NOT write any math symbols or equations using LaTeX delimiters like \\[, \\], \\(, \\), or double/single dollar signs ($). Write all mathematical equations and formulas in simple, plain english text representation.
"""
                    if is_active:
                        agent2_output = call_provider_llm_api(llm_provider, selected_model, prompt_a2, gemini_api_key, ollama_url)
                    else:
                        excess = backtest_results.get('strategy_return_pct', 0.0) - backtest_results.get('benchmark_return_pct', 0.0)
                        agent2_output = f"""
- **Strategy Return**: The backtested trading strategy generated **{backtest_results.get('strategy_return_pct', 0.0):.2f}%** return vs **{backtest_results.get('benchmark_return_pct', 0.0):.2f}%** for simple Buy & Hold.
- **Excess Return**: The strategy beat the benchmark by **{excess:+.2f}%**.
- **Risk Metrics**: Max Drawdown was capped at **{backtest_results.get('max_drawdown_pct', 0.0):.2f}%** with annualized volatility of **{backtest_results.get('volatility_pct', 0.0):.2f}%**.
- **Sentiment Scores**: Ticker sentiment compound score is **{avg_sentiment:+.2f}**, and industry/sector query sentiment is **{avg_industry_sentiment:+.2f}**.
- **Trading Simulator Comparison**: The Active Multi-Trade Simulator's scale-in/scale-out approach generated **{multi_trade_results.get('strategy_return_pct', 0.0):.2f}%** return, highlighting the mathematical advantages of capital scaling over binary entry.
"""

                # Synthesis & Consensus Engine
                with st.spinner("Invoking Synthesis & Consensus Engine..."):
                    prompt_a3 = f"""
You are the Synthesis & Consensus Engine. Act as the Chief Investment Officer to synthesize findings and generate a final rating consensus for {ticker}.

Active Multi-Trade Simulator Data:
---
{agent5_output}
---

Technical Inputs:
---
{agent1_output}
---

Quantitative Inputs:
---
{agent2_output}
---

Instructions:
1. Synthesize chart indicators, quantitative metrics, and active multi-trade outcomes to produce a complete consensus report.
2. State a bold **Consensus Rating**: [Strong Buy / Buy / Hold / Sell / Strong Sell], followed by key bulleted rationale.
3. Integrate Trading Agent Data: Address how the active trading returns support or contradict the technical signals and statistical metrics to reach a final result.
4. Explain the consensus using simple concepts (explaining conflicts or confirmations clearly) but in professional, adult-appropriate language.
5. Structure & Formatting Requirements:
   - Output in clean, highly structured Markdown.
   - Use bold subheadings (e.g., ### Section Name) for logical sections.
   - Use bullet points with bold key metrics.
   - Separate distinct ideas with spacing or horizontal rules (`---`).
   - Do not output any HTML tags.
   - Start directly with the structured answer (no conversational preamble).
   - IMPORTANT: DO NOT write any math symbols or equations using LaTeX delimiters like \\[, \\], \\(, \\), or double/single dollar signs ($). Write all mathematical equations and formulas in simple, plain english text representation.
"""
                    if is_active:
                        agent3_output = call_provider_llm_api(llm_provider, selected_model, prompt_a3, gemini_api_key, ollama_url)
                    else:
                        agent3_output = f"""
- **Consensus Rating**: **{latest_rec.get('action', 'HOLD')}** (Consensus Score: **{latest_rec.get('score', 0.0):+.2f}**)
- **Consensus Findings**:
  - The technical charts suggest a momentum profile of **{latest_rsi:.1f}** with positive/negative MACD acceleration.
  - The backtest validates this with a Strategy Return of **{backtest_results.get('strategy_return_pct', 0.0):.1f}%** for all-in/all-out.
  - The Active Multi-Trade Simulator generated a return of **{multi_trade_results.get('strategy_return_pct', 0.0):.1f}%** across **{multi_trade_results.get('total_trades', 0)}** active executions.
  - Combining these insights, our consensus panel rates the stock as a **{latest_rec.get('action', 'HOLD')}**.
"""
                
                # Portfolio Allocation Advisor
                with st.spinner("Invoking Portfolio Allocation Advisor..."):
                    COMPETITORS_MAP = {
                        "AAPL": ["MSFT", "GOOGL", "NVDA", "AMZN"],
                        "MSFT": ["AAPL", "GOOGL", "AMZN", "META"],
                        "GOOG": ["GOOGL", "MSFT", "META", "AMZN"],
                        "GOOGL": ["GOOG", "MSFT", "META", "AMZN"],
                        "TSLA": ["BYDDF", "F", "GM", "NIO"],
                        "NVDA": ["AMD", "INTC", "TSMC", "QCOM"],
                        "AMD": ["NVDA", "INTC", "QCOM", "ARM"],
                        "AMZN": ["WMT", "EBAY", "TGT", "BABA"],
                        "META": ["GOOGL", "SNAP", "PINS", "MSFT"],
                        "NFLX": ["DIS", "WBD", "PARA", "CMCSA"],
                        "JPM": ["BAC", "WFC", "C", "GS"],
                        "V": ["MA", "AXP", "PYPL", "DFS"]
                    }
                    peers = COMPETITORS_MAP.get(ticker.upper(), ["SPY", "QQQ", "IWM"])
                    prompt_a4 = f"""
You are the Portfolio Allocation Advisor. Review the consensus synthesis and recommend the capital allocation profile for {ticker}.

Active Multi-Trade Simulator Data:
---
{agent5_output}
---

Consensus Synthesis:
---
{agent3_output}
---

Investor Context:
- Capital: ${starting_capital:,.2f}
- Ticker: {ticker} | Peers: {", ".join(peers)}

Instructions:
1. Provide a complete, fully formed comparison, allocation rating, and entry strategy.
2. Integrate Trading Agent Data: Use the active multi-trade outcomes and entry log statistics to optimize the sizing, average cost target, and entry trigger prices.
3. Outline **Alternative Analysis** (competitors comparison) and **Asset Allocation** (sizing and entry trigger prices) in simple terms (e.g. entry compared to shopping on a discount, or sizing to not putting all eggs in one basket).
4. Keep the tone professional for adults but extremely easy to understand.
5. Structure & Formatting Requirements:
   - Output in clean, highly structured Markdown.
   - Use bold subheadings (e.g., ### Section Name) for logical sections.
   - Use bullet points with bold key metrics.
   - Separate distinct ideas with spacing or horizontal rules (`---`).
   - Do not output any HTML tags.
   - Start directly with the structured answer (no conversational preamble).
   - IMPORTANT: DO NOT write any math symbols or equations using LaTeX delimiters like \\[, \\], \\(, \\), or double/single dollar signs ($). Write all mathematical equations and formulas in simple, plain english text representation.
"""
                    if is_active:
                        agent4_output = call_provider_llm_api(llm_provider, selected_model, prompt_a4, gemini_api_key, ollama_url)
                    else:
                        rec_price = latest_row.get('SMA_20', latest_close)
                        rec_alloc = starting_capital * 0.15
                        agent4_output = f"""
1. **Competitive Analysis**: We compared {ticker} against its primary sector peers: **{", ".join(peers)}**.
2. **Alternative Recommendations**:
   - If looking for pure momentum, sector leaders like **{peers[0]}** or **{peers[1]}** may offer competitive volatility profiles.
   - If {ticker} fits your value/quant criteria, it remains a robust choice.
3. **Allocation Sizing**:
   - **Recommended Entry Price**: **${rec_price:,.2f}** (aligned with its 20-day Simple Moving Average boundary).
   - **Suggested Capital Allocation**: **15%** of your starting capital, which equates to **${rec_alloc:,.2f}** from your total **${starting_capital:,.2f}**.
- **Trading Simulator Integration**: Based on the active multi-trade simulations, we suggest using 25% tranches for entry to average down purchase costs, targeting the **${rec_price:,.2f}** zone.
"""

                # Save results to session state
                st.session_state[ma_state_key] = {
                    "agent1": agent1_output,
                    "agent2": agent2_output,
                    "agent3": agent3_output,
                    "agent4": agent4_output,
                    "agent5": agent5_output
                }
            
            # Display results
            results = st.session_state[ma_state_key]
            
            # Display results vertically
            st.markdown("""
            <div style="background: rgba(30, 30, 60, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border-left: 5px solid #ff4b2b;">
                <h4 style="color: #ff4b2b; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">⚡ Active Multi-Trade Simulator</h4>
            """, unsafe_allow_html=True)
            st.markdown(results["agent5"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div style="background: rgba(30, 30, 60, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border-left: 5px solid #00f2fe;">
                <h4 style="color: #00f2fe; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">📊 Chart Intelligence Analyst</h4>
            """, unsafe_allow_html=True)
            st.markdown(results["agent1"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(30, 30, 60, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border-left: 5px solid #ffd200;">
                <h4 style="color: #ffd200; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">🔢 Quantitative Metrics Parser</h4>
            """, unsafe_allow_html=True)
            st.markdown(results["agent2"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(30, 30, 60, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border-left: 5px solid #a18cd1;">
                <h4 style="color: #a18cd1; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">🧠 Synthesis & Consensus Engine</h4>
            """, unsafe_allow_html=True)
            st.markdown(results["agent3"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(30, 30, 60, 0.2); border: 1px solid rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; border-left: 5px solid #00b09b;">
                <h4 style="color: #00b09b; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">🎯 Portfolio Allocation Advisor</h4>
            """, unsafe_allow_html=True)
            st.markdown(results["agent4"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Actionable directive TL;DR card at the very bottom
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3f35 0%, #112217 100%); border: 2px solid #00b09b; padding: 2rem; border-radius: 20px; margin-top: 2.5rem; margin-bottom: 2rem; border-left: 8px solid #00b09b; box-shadow: 0 8px 32px 0 rgba(0, 176, 155, 0.25);">
                <h3 style="color: #00f2fe; margin-top: 0; display: flex; align-items: center; gap: 0.5rem; font-weight: 800;">🎯 Actionable TL;DR Consensus Directive</h3>
                <p style="color: #c9d1d9; font-size: 1.05rem; margin-bottom: 1.5rem;">
                    Consolidated strategic summary of the multi-agent consensus panel analysis for <b>{ticker}</b>.
                </p>
                <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
                    <div style="flex: 1; min-width: 200px; background: rgba(0,0,0,0.25); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.85rem; color: #8f9cae; font-weight: bold; text-transform: uppercase;">Consensus Action</span>
                        <h4 style="margin: 5px 0 0 0; color: #00b09b; font-size: 1.6rem; font-weight: 800;">{action}</h4>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(0,0,0,0.25); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.85rem; color: #8f9cae; font-weight: bold; text-transform: uppercase;">Active Strategy Return</span>
                        <h4 style="margin: 5px 0 0 0; color: #ffd200; font-size: 1.6rem; font-weight: 800;">{multi_trade_results.get('strategy_return_pct', 0.0):+.2f}%</h4>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(0,0,0,0.25); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.85rem; color: #8f9cae; font-weight: bold; text-transform: uppercase;">Target Entry Price (SMA)</span>
                        <h4 style="margin: 5px 0 0 0; color: #00f2fe; font-size: 1.6rem; font-weight: 800;">${latest_row.get('SMA_20', latest_close):,.2f}</h4>
                    </div>
                </div>
                <div style="color: #c9d1d9; font-size: 0.95rem; line-height: 1.6;">
                    <b>Core Directive:</b> Technical indicators, risk-reward ratios, sentiment indexes, and the Active Multi-Trade Simulation have converged. 
                    Refer to the <b>Synthesis & Consensus Engine</b> for the consensus analysis details and the <b>Portfolio Allocation Advisor</b> for the portfolio allocation and capital distribution sizing limits.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to clear cache and rerun
            if st.button("🔄 Clear & Re-Run Consensus Analysis"):
                if ma_state_key in st.session_state:
                    del st.session_state[ma_state_key]
                st.rerun()
                
        else:
            st.info("💡 Click the button above to invoke the Multi-Agent Consensus Panel. This runs a deep assessment of all technical charts, metrics, and risk factors across four specialized quantitative and advisory agents.")

    # Real-time refresh loop trigger
    if live_streaming:
        import time
        time.sleep(1)
        st.rerun()

except Exception as e:
    st.error(f"An unexpected error occurred during processing: {e}")
    st.info("Please review the settings sidebar and try again.")
