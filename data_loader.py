import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Downloads historical market data for a given ticker from Yahoo Finance.
    
    Parameters:
    - ticker: Ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
    - period: Data period to download (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
    - interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
    - pd.DataFrame containing columns: Open, High, Low, Close, Adj Close, Volume
    """
    print(f"Fetching historical price data for {ticker} (period={period}, interval={interval})...")
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval)
    return df

def fetch_stock_news(ticker: str) -> list:
    """
    Fetches the latest news headlines for the given ticker from Yahoo Finance.
    
    Returns:
    - A list of dicts, where each dict has: 'title', 'publisher', 'link', 'related_tickers'
    """
    print(f"Fetching recent news headlines for {ticker}...")
    ticker_obj = yf.Ticker(ticker)
    news_items = ticker_obj.news
    
    processed_news = []
    if news_items:
        for item in news_items:
            processed_news.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', ''),
                'providerPublishTime': item.get('providerPublishTime', 0),
                'related_tickers': item.get('relatedTickers', [])
            })
    return processed_news

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def fetch_company_info(ticker: str) -> dict:
    """
    Fetches company profile details (long name, sector, industry, summary) via yfinance.
    """
    print(f"Fetching company info for {ticker}...")
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'summary': info.get('longBusinessSummary', '')
        }
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return {
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'summary': ''
        }

def fetch_industry_news(query: str) -> list:
    """
    Fetches the latest news headlines from Google News RSS feed for a given query.
    """
    print(f"Fetching Google News RSS for query: {query}...")
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req, timeout=10).read()
        root = ET.fromstring(xml_data)
        
        news_items = []
        for item in root.findall('.//item')[:15]:  # Limit to top 15 news items
            title = item.find('title').text if item.find('title') is not None else ''
            link = item.find('link').text if item.find('link') is not None else ''
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
            
            # Extract publisher from title (usually suffix " - Publisher Name")
            publisher = 'Google News'
            if ' - ' in title:
                parts = title.split(' - ')
                title = ' - '.join(parts[:-1])
                publisher = parts[-1]
                
            news_items.append({
                'title': title,
                'publisher': publisher,
                'link': link,
                'pubDate': pub_date
            })
        return news_items
    except Exception as e:
        print(f"Error fetching Google News RSS: {e}")
        return []

if __name__ == '__main__':
    # Simple test run
    df = fetch_stock_data("AAPL", period="1mo")
    print(df.tail())
    
    info = fetch_company_info("AAPL")
    print(f"\nCompany Info: {info['name']} | Sector: {info['sector']} | Industry: {info['industry']}")
    
    news = fetch_stock_news("AAPL")
    print(f"\nFetched {len(news)} news headlines. Example headline:")
    if news:
        print(f"- {news[0]['title']} (by {news[0]['publisher']})")
        
    ind_news = fetch_industry_news(f"{info['sector']} trends")
    print(f"\nFetched {len(ind_news)} industry news headlines. Example headline:")
    if ind_news:
        print(f"- {ind_news[0]['title']} (by {ind_news[0]['publisher']})")
