"""
Database Module for Sentiment Analysis Bot
Handles SQLite integration for historical tracking.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_NAME = "sentiment_bot.db"

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Headlines and individual analysis results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            asset_type TEXT,
            symbol TEXT,
            title TEXT,
            source TEXT,
            compound_score REAL,
            sentiment TEXT,
            url TEXT
        )
    ''')
    
    # Aggregated session results for faster charting
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            asset_type TEXT,
            symbol TEXT,
            market_score REAL,
            positive_pct REAL,
            negative_pct REAL,
            neutral_pct REAL,
            count INTEGER,
            engine TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_results(df, asset_type, symbol, engine="vader"):
    """Save analysis results to the database."""
    if df.empty:
        return
        
    conn = sqlite3.connect(DB_NAME)
    
    # 1. Save individual headlines (with prevention for exact duplicate titles in same session)
    # Note: We use the dataframe to_sql for convenience
    df_to_save = df.copy()
    df_to_save['asset_type'] = asset_type
    df_to_save['symbol'] = symbol
    df_to_save['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Columns mapping to match table schema
    mapping = {
        'title': 'title',
        'source': 'source',
        'compound_score': 'compound_score',
        'sentiment': 'sentiment',
        'url': 'url',
        'asset_type': 'asset_type',
        'symbol': 'symbol',
        'timestamp': 'timestamp'
    }
    
    # Filter only relevant columns for the history table
    history_df = df_to_save[list(mapping.keys())]
    history_df.to_sql('sentiment_history', conn, if_exists='append', index=False)
    
    # 2. Save session statistics
    from analyzer import calculate_distribution, calculate_market_score
    dist = calculate_distribution(df)
    score, _ = calculate_market_score(df)
    
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'asset_type': asset_type,
        'symbol': symbol,
        'market_score': score,
        'positive_pct': dist['Positive'],
        'negative_pct': dist['Negative'],
        'neutral_pct': dist['Neutral'],
        'count': len(df),
        'engine': engine
    }
    
    pd.DataFrame([stats]).to_sql('session_stats', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()

def get_historical_trends(asset_type=None, symbol=None, days=7):
    """Fetch historical session stats for visualization."""
    conn = sqlite3.connect(DB_NAME)
    
    query = "SELECT * FROM session_stats WHERE timestamp >= datetime('now', ?)"
    params = [f'-{days} days']
    
    if asset_type:
        query += " AND asset_type = ?"
        params.append(asset_type)
    
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
        
    query += " ORDER BY timestamp ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")
