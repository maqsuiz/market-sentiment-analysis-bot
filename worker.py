"""
Market Sentiment Bot - Background Worker
Automates crypto and BIST100 analysis on a schedule.
"""

import schedule
import time
import logging
from datetime import datetime
import os
import sys

# Import bot functions
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment_bot import run_sentiment_analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("worker.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def job():
    """Analysis task to be run on schedule."""
    logging.info("--- Starting Scheduled Analysis Job ---")
    
    # 1. Analyze Bitcoin
    try:
        logging.info("Analyzing Bitcoin...")
        run_sentiment_analysis(asset_type="crypto", symbol="bitcoin", count=30)
        logging.info("Bitcoin analysis successful.")
    except Exception as e:
        logging.error(f"Bitcoin analysis failed: {e}")
        
    # 2. Analyze BIST100
    try:
        logging.info("Analyzing BIST100...")
        run_sentiment_analysis(asset_type="bist", count=30)
        logging.info("BIST100 analysis successful.")
    except Exception as e:
        logging.error(f"BIST100 analysis failed: {e}")
        
    logging.info("--- Scheduled Analysis Job Complete ---")

def main():
    logging.info("Market Sentiment Bot Worker Started.")
    logging.info("Schedule: Every 1 hour.")
    
    # Run once immediately on start
    job()
    
    # Setup hourly schedule
    schedule.every(1).hours.do(job)
    
    logging.info("Worker is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Worker stopped by user.")

if __name__ == "__main__":
    main()
