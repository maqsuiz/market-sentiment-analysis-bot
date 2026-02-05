"""
Data Collection Module for Sentiment Analysis Bot
Fetches news headlines for cryptocurrencies and BIST100 companies.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime
from typing import List, Dict, Optional


# Headers to avoid being blocked by websites
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_crypto_news(symbol: str = "bitcoin", count: int = 100) -> List[Dict]:
    """
    Fetch cryptocurrency news headlines from multiple sources.
    
    Args:
        symbol: Cryptocurrency name (e.g., 'bitcoin', 'ethereum')
        count: Maximum number of headlines to fetch
        
    Returns:
        List of dictionaries with 'title', 'source', 'date', 'url' keys
    """
    headlines = []
    
    # Source 1: CoinDesk RSS Feed
    try:
        coindesk_headlines = _fetch_coindesk_rss(symbol, count // 3)
        headlines.extend(coindesk_headlines)
    except Exception as e:
        print(f"[WARNING] CoinDesk fetch failed: {e}")
    
    # Source 2: CryptoNews RSS
    try:
        cryptonews_headlines = _fetch_cryptonews_rss(symbol, count // 3)
        headlines.extend(cryptonews_headlines)
    except Exception as e:
        print(f"[WARNING] CryptoNews fetch failed: {e}")
    
    # Source 3: Google News RSS (crypto search)
    try:
        google_headlines = _fetch_google_news_rss(f"{symbol} cryptocurrency", count // 3)
        headlines.extend(google_headlines)
    except Exception as e:
        print(f"[WARNING] Google News fetch failed: {e}")
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_headlines = []
    for item in headlines:
        title_lower = item['title'].lower().strip()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_headlines.append(item)
    
    return unique_headlines[:count]


def _fetch_coindesk_rss(symbol: str, count: int) -> List[Dict]:
    """Fetch from CoinDesk RSS feed."""
    rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    
    feed = feedparser.parse(rss_url)
    headlines = []
    
    symbol_lower = symbol.lower()
    for entry in feed.entries[:count * 2]:  # Fetch extra to filter
        title = entry.get('title', '')
        if symbol_lower in title.lower() or symbol_lower == 'bitcoin':
            headlines.append({
                'title': title,
                'source': 'CoinDesk',
                'date': entry.get('published', datetime.now().isoformat()),
                'url': entry.get('link', '')
            })
        if len(headlines) >= count:
            break
    
    return headlines


def _fetch_cryptonews_rss(symbol: str, count: int) -> List[Dict]:
    """Fetch from CryptoNews RSS feed."""
    rss_url = "https://cryptonews.com/news/feed/"
    
    feed = feedparser.parse(rss_url)
    headlines = []
    
    symbol_lower = symbol.lower()
    for entry in feed.entries[:count * 2]:
        title = entry.get('title', '')
        if symbol_lower in title.lower() or len(headlines) < count // 2:
            headlines.append({
                'title': title,
                'source': 'CryptoNews',
                'date': entry.get('published', datetime.now().isoformat()),
                'url': entry.get('link', '')
            })
        if len(headlines) >= count:
            break
    
    return headlines


def _fetch_google_news_rss(query: str, count: int) -> List[Dict]:
    """Fetch from Google News RSS search."""
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    headlines = []
    
    for entry in feed.entries[:count]:
        headlines.append({
            'title': entry.get('title', ''),
            'source': 'Google News',
            'date': entry.get('published', datetime.now().isoformat()),
            'url': entry.get('link', '')
        })
    
    return headlines


def fetch_bist_news(count: int = 100) -> List[Dict]:
    """
    Fetch BIST100 (Borsa Istanbul) related news headlines.
    
    Args:
        count: Maximum number of headlines to fetch
        
    Returns:
        List of dictionaries with 'title', 'source', 'date', 'url' keys
    """
    headlines = []
    
    # Source 1: Google News RSS for BIST100
    try:
        bist_google = _fetch_google_news_rss("BIST100 borsa istanbul", count // 2)
        headlines.extend(bist_google)
    except Exception as e:
        print(f"[WARNING] BIST Google News fetch failed: {e}")
    
    # Source 2: Search for Turkish market news
    try:
        turkey_google = _fetch_google_news_rss("Turkey stock market economy", count // 2)
        headlines.extend(turkey_google)
    except Exception as e:
        print(f"[WARNING] Turkey news fetch failed: {e}")
    
    # Remove duplicates
    seen_titles = set()
    unique_headlines = []
    for item in headlines:
        title_lower = item['title'].lower().strip()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_headlines.append(item)
    
    return unique_headlines[:count]


def fetch_headlines(asset_type: str = "crypto", symbol: str = "bitcoin", count: int = 100) -> List[Dict]:
    """
    Main function to fetch headlines based on asset type.
    
    Args:
        asset_type: Either 'crypto' or 'bist'
        symbol: For crypto, the cryptocurrency name (e.g., 'bitcoin')
        count: Maximum number of headlines
        
    Returns:
        List of headline dictionaries
    """
    if asset_type.lower() == "crypto":
        return fetch_crypto_news(symbol, count)
    elif asset_type.lower() == "bist":
        return fetch_bist_news(count)
    else:
        raise ValueError(f"Unknown asset type: {asset_type}. Use 'crypto' or 'bist'.")


if __name__ == "__main__":
    # Quick test
    print("Testing crypto news fetch...")
    news = fetch_crypto_news("bitcoin", 10)
    for i, item in enumerate(news, 1):
        print(f"{i}. [{item['source']}] {item['title'][:70]}...")
    print(f"\nTotal fetched: {len(news)} headlines")
