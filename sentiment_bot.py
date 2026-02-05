"""
Sentiment Analysis Bot for Financial Markets
Measures market psychology by analyzing news headlines for cryptocurrencies and BIST100.

Author: Sentiment Bot
Version: 1.0.0
"""

import argparse
import pandas as pd
from datetime import datetime

from scraper import fetch_headlines
from analyzer import (
    analyze_headlines,
    calculate_distribution,
    calculate_market_score,
    get_extreme_headlines
)
from validator import evaluate_accuracy, print_evaluation_report
import database


def print_banner():
    """Print application banner."""
    banner = """
+===============================================================+
|       MARKET SENTIMENT ANALYSIS BOT v1.0                      |
|       Measuring Market Psychology Through News Analysis       |
+===============================================================+
    """
    print(banner)


def print_results(df: pd.DataFrame, distribution: dict, score: float, interpretation: str):
    """
    Print formatted analysis results.
    
    Args:
        df: DataFrame with sentiment analysis results
        distribution: Sentiment distribution percentages
        score: Overall market sentiment score
        interpretation: Score interpretation text
    """
    print("\n" + "="*70)
    print("HEADLINE ANALYSIS RESULTS")
    print("="*70)
    
    # Configure pandas display
    pd.set_option('display.max_colwidth', 60)
    pd.set_option('display.width', None)
    
    # Show summary table
    display_df = df[['title', 'source', 'compound_score', 'sentiment']].copy()
    display_df.columns = ['Headline', 'Source', 'Score', 'Sentiment']
    print(display_df.to_string(index=True))
    
    # Sentiment Distribution
    print("\n" + "="*70)
    print("SENTIMENT DISTRIBUTION")
    print("="*70)
    
    total = len(df)
    for sentiment, pct in distribution.items():
        count = int(pct * total / 100)
        bar_length = int(pct / 2)  # Scale bar to 50 chars max
        
        if sentiment == "Positive":
            bar = "#" * bar_length
            emoji = "[+]"
        elif sentiment == "Negative":
            bar = "#" * bar_length
            emoji = "[-]"
        else:
            bar = "#" * bar_length
            emoji = "[=]"
        
        print(f"{emoji} {sentiment:<10}: {bar} {pct}% ({count} headlines)")
    
    # Market Score
    print("\n" + "="*70)
    print("OVERALL MARKET SENTIMENT")
    print("="*70)
    
    # Visual score bar
    score_normalized = (score + 100) / 2  # Convert -100,100 to 0,100
    bar_position = int(score_normalized / 2)
    
    scale = "Bearish <" + "-" * 20 + "|" + "-" * 20 + "> Bullish"
    print(f"\n{scale}")
    
    pointer = " " * (bar_position + 9) + "^"
    print(pointer)
    
    print(f"\n   Market Score: {score:+.2f}")
    print(f"   Interpretation: {interpretation}")
    
    # Most extreme headlines
    print("\n" + "="*70)
    print("MOST BULLISH HEADLINES")
    print("-"*70)
    most_positive, most_negative = get_extreme_headlines(df, n=3)
    for _, row in most_positive.iterrows():
        print(f"  [{row['compound_score']:+.3f}] {row['title'][:60]}...")
    
    print("\nMOST BEARISH HEADLINES")
    print("-"*70)
    for _, row in most_negative.iterrows():
        print(f"  [{row['compound_score']:+.3f}] {row['title'][:60]}...")


def run_sentiment_analysis(asset_type: str, symbol: str = "bitcoin", count: int = 100, 
                           validate: bool = False, save_csv: bool = False):
    """
    Main function to run the sentiment analysis pipeline.
    
    Args:
        asset_type: 'crypto' or 'bist'
        symbol: For crypto, the cryptocurrency name
        count: Number of headlines to fetch
        validate: Whether to run accuracy validation
        save_csv: Whether to save results to CSV
    """
    print_banner()
    
    print(f"\nFetching {count} headlines for {symbol.upper() if asset_type == 'crypto' else 'BIST100'}...")
    print(f"   Asset Type: {asset_type.upper()}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Fetch headlines
    try:
        headlines = fetch_headlines(asset_type, symbol, count)
        print(f"   [OK] Successfully fetched {len(headlines)} headlines")
    except Exception as e:
        print(f"   [ERROR] Error fetching headlines: {e}")
        return None
    
    if len(headlines) == 0:
        print("\n[WARNING] No headlines found. Please try again later or check network connection.")
        return None
    
    # Step 2: Analyze sentiment
    print("\nAnalyzing sentiment...")
    df = analyze_headlines(headlines)
    print(f"   [OK] Analysis complete")
    
    # Step 3: Calculate metrics
    distribution = calculate_distribution(df)
    score, interpretation = calculate_market_score(df)
    
    # Step 4: Display results
    print_results(df, distribution, score, interpretation)
    
    # Step 5: Save to Database (New in Technical Infra update)
    try:
        database.init_db()
        database.save_results(df, asset_type, symbol)
        print("\n   [DB] Results saved to historical database")
    except Exception as e:
        print(f"\n   [DB ERROR] Could not save to database: {e}")
    
    # Step 6: Save to CSV if requested
    if save_csv:
        filename = f"sentiment_results_{asset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
    
    # Step 6: Run validation if requested
    if validate:
        print("\n" + "="*70)
        print("RUNNING MODEL VALIDATION")
        print("="*70)
        print_evaluation_report()
    
    return df, distribution, score, interpretation


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Market Sentiment Analysis Bot - Analyze news for market psychology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentiment_bot.py --asset crypto --symbol bitcoin --count 50
  python sentiment_bot.py --asset bist --count 30
  python sentiment_bot.py --asset crypto --symbol ethereum --validate
  python sentiment_bot.py --asset crypto --save-csv
        """
    )
    
    parser.add_argument(
        '--asset', '-a',
        type=str,
        choices=['crypto', 'bist'],
        default='crypto',
        help='Asset type to analyze (default: crypto)'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='bitcoin',
        help='Cryptocurrency symbol for crypto analysis (default: bitcoin)'
    )
    
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=100,
        help='Number of headlines to fetch (default: 100)'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Run accuracy validation on test set'
    )
    
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Save results to CSV file'
    )
    
    args = parser.parse_args()
    
    run_sentiment_analysis(
        asset_type=args.asset,
        symbol=args.symbol,
        count=args.count,
        validate=args.validate,
        save_csv=args.save_csv
    )


if __name__ == "__main__":
    main()
