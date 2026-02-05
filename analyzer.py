"""
Sentiment Analysis Module for Market Psychology Bot
Supports multiple engines: VADER, FinBERT, and Gemini Pro.
"""

import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
try:
    from google import genai
except ImportError:
    genai = None
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- VADER Initialization ---
vader_analyzer = SentimentIntensityAnalyzer()
FINANCIAL_LEXICON = {
    'bullish': 2.5, 'moon': 2.0, 'mooning': 2.5, 'pump': 1.5, 'rally': 2.0,
    'surge': 2.0, 'soar': 2.0, 'breakout': 1.8, 'all-time high': 2.5,
    'ath': 2.0, 'hodl': 1.5, 'accumulate': 1.0, 'upgrade': 1.5,
    'outperform': 1.5, 'adoption': 1.5, 'institutional': 1.0,
    'bearish': -2.5, 'crash': -3.0, 'dump': -2.0, 'plunge': -2.5,
    'collapse': -3.0, 'fear': -2.0, 'panic': -2.5, 'sell-off': -2.0,
    'selloff': -2.0, 'liquidation': -2.0, 'downgrade': -1.5,
    'underperform': -1.5, 'banned': -2.0, 'hack': -2.5, 'scam': -3.0,
    'fraud': -3.0, 'bubble': -1.5, 'fud': -1.5, 'warning': -1.5, 'risk': -1.0,
}
vader_analyzer.lexicon.update(FINANCIAL_LEXICON)

# --- FinBERT Initialization (Lazy Loading) ---
_finbert_pipe = None

def get_finbert():
    global _finbert_pipe
    if _finbert_pipe is None:
        # Load the FinBERT model
        _finbert_pipe = pipeline("text-classification", model="ProsusAI/finbert")
    return _finbert_pipe

# --- Gemini Initialization ---
def setup_gemini(api_key: Optional[str] = None):
    """Sets up Gemini using the new google-genai SDK (v1.8)."""
    if genai is None:
        print("[ERROR] google-genai package not found. Run 'pip install google-genai'.")
        return None
        
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
        
    try:
        client = genai.Client(api_key=key)
        
        # Priority list for 2026
        priority = [
            'gemini-2.0-flash', 
            'gemini-2.0-pro',
            'gemini-1.5-pro',
            'gemini-1.5-flash-002',
            'gemini-1.5-flash', 
            'gemini-pro'
        ]
        
        # Discover available models using the new API
        available = []
        try:
            for m in client.models.list():
                if 'generateContent' in m.supported_generation_methods:
                    available.append(m.name)
        except Exception:
            available = [p for p in priority]

        # Scan models and return the first one that passes a LIVE PING
        for p in priority:
            # Check if p is in available or matches exactly
            match = next((am for am in available if p in am), None)
            if match:
                try:
                    # Connectivity Ping using the new client
                    client.models.generate_content(
                        model=match,
                        contents="ping",
                        config={"max_output_tokens": 1}
                    )
                    # We store the selected model name on the client for convenience
                    client.selected_model = match
                    return client # Confirmed working
                except Exception:
                    continue 
        
        return None 
    except Exception as e:
        print(f"Gemini critical setup failure: {e}")
        return None

def analyze_vader(text: str) -> Dict:
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    sentiment = 'Positive' if compound >= 0.05 else 'Negative' if compound <= -0.05 else 'Neutral'
    return {
        'score': round(compound, 4),
        'sentiment': sentiment,
        'reasoning': "Rule-based analysis using financial lexicon."
    }

def analyze_finbert(text: str) -> Dict:
    pipe = get_finbert()
    result = pipe(text)[0]
    label = result['label'].capitalize()
    # Map FinBERT labels to scores roughly
    score_map = {'Positive': 0.8, 'Negative': -0.8, 'Neutral': 0.0}
    return {
        'score': score_map.get(label, 0.0),
        'sentiment': label,
        'reasoning': f"LLM-based financial model (Confidence: {result['score']:.2f})"
    }

def analyze_gemini(text: str, client) -> Dict:
    if not client:
        return analyze_vader(text)
    
    prompt = f"""
    Analyze the sentiment of this financial headline and provide a brief reasoning:
    Headline: "{text}"
    
    Response MUST follow this EXACT format:
    Sentiment: [Positive/Negative/Neutral]
    Score: [Between -1.0 and 1.0]
    Reasoning: [One sentence explanation]
    """
    try:
        response = client.models.generate_content(
            model=client.selected_model,
            contents=prompt
        )
        content = response.text.strip()
        
        # Robust parsing
        sentiment = "Neutral"
        score = 0.0
        reasoning = f"AI Analysis ({client.selected_model})"
        
        for line in content.split('\n'):
            line = line.strip()
            if line.lower().startswith('sentiment:'):
                sentiment = line.split(':', 1)[1].strip().capitalize()
            elif line.lower().startswith('score:'):
                try:
                    val = "".join(c for c in line.split(':', 1)[1] if c.isdigit() or c in '.-')
                    score = float(val)
                except: pass
            elif line.lower().startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            sentiment = 'Neutral'
            
        return {
            'score': score,
            'sentiment': sentiment,
            'reasoning': reasoning
        }
    except Exception as e:
        res = analyze_vader(text)
        error_msg = str(e).lower()
        model_name = getattr(client, 'selected_model', 'Unknown')
        if "404" in error_msg or "not found" in error_msg:
            res['reasoning'] = f"AI Model Retired ({model_name}). Using VADER fallback."
        else:
            res['reasoning'] = f"AI Error ({model_name}): {str(e)[:40]}..."
        return res

def analyze_headlines(headlines: List[Dict], engine: str = 'vader', api_key: Optional[str] = None) -> pd.DataFrame:
    results = []
    client = None
    
    if engine == 'gemini':
        client = setup_gemini(api_key)
    
    for item in headlines:
        title = item.get('title', '')
        
        if engine == 'finbert':
            analysis = analyze_finbert(title)
        elif engine == 'gemini':
            analysis = analyze_gemini(title, client)
        else:
            analysis = analyze_vader(title)
            
        results.append({
            'title': title,
            'source': item.get('source', 'Unknown'),
            'date': item.get('date', ''),
            'url': item.get('url', ''),
            'compound_score': analysis['score'],
            'sentiment': analysis['sentiment'],
            'reasoning': analysis['reasoning'],
            'topic': categorize_topic(title)
        })
    
    return pd.DataFrame(results)

def calculate_distribution(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 0.0}
    counts = df['sentiment'].value_counts()
    total = len(df)
    return {
        'Positive': round((counts.get('Positive', 0) / total) * 100, 2),
        'Negative': round((counts.get('Negative', 0) / total) * 100, 2),
        'Neutral': round((counts.get('Neutral', 0) / total) * 100, 2)
    }

def categorize_topic(text: str) -> str:
    """Categorize the headline into a market topic."""
    text = text.lower()
    
    topics = {
        'Regulation': ['sec', 'regulation', 'law', 'legal', 'tax', 'ban', 'court', 'case', 'etf', 'approval', 'policy'],
        'Macro': ['fed', 'inflation', 'cpi', 'interest', 'bank', 'recession', 'economy', 'growth', 'unemployment', 'debt'],
        'Security': ['hack', 'scam', 'drain', 'exploit', 'phishing', 'security', 'vulnerability', 'attack', 'heist'],
        'Tech': ['upgrade', 'fork', 'mainnet', 'testnet', 'protocol', 'development', 'node', 'hashrate', 'mining'],
        'Market': ['price', 'surge', 'drop', 'plunge', 'dip', 'high', 'low', 'bull', 'bear', 'rally', 'whale', 'crash']
    }
    
    for topic, keywords in topics.items():
        if any(kw in text for kw in keywords):
            return topic
            
    return 'General'


def calculate_market_score(df: pd.DataFrame) -> Tuple[float, str]:
    """Calculate an overall sentiment score and interpretation."""
    if df.empty: return 0.0, "No data"
    avg_score = df['compound_score'].mean() * 100
    
    if avg_score >= 60:
        interp = "Extreme Greed"
    elif avg_score >= 20:
        interp = "Bullish"
    elif avg_score <= -60:
        interp = "Extreme Fear"
    elif avg_score <= -20:
        interp = "Bearish"
    else:
        interp = "Neutral"
        
    return avg_score, interp

def get_extreme_headlines(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values('compound_score', ascending=False)
    return sorted_df.head(n), sorted_df.tail(n)

def generate_executive_summary(df: pd.DataFrame, client) -> str:
    """
    Generate a 3-sentence Turkish executive summary using LLM.
    """
    if not client or df.empty:
        return "Summary could not be generated. Please check your API key or ensure data is available."
    
    # Select top 5 positive and 5 negative headlines
    pos, neg = get_extreme_headlines(df, n=5)
    
    headlines_text = "\nPositive Headlines:\n"
    headlines_text += "\n".join([f"- {row['title']}" for _, row in pos.iterrows()])
    headlines_text += "\nNegative Headlines:\n"
    headlines_text += "\n".join([f"- {row['title']}" for _, row in neg.iterrows()])
    
    prompt = f"""
    Based on the following financial news headlines, write a professional and informative 
    3-sentence Executive Summary that captures the current market psychology.
    
    Headlines:
    {headlines_text}
    
    The response must be exactly 3 sentences of English text.
    """
    
    try:
        response = client.models.generate_content(
            model=client.selected_model,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {str(e)[:100]}"

if __name__ == "__main__":
    test_headlines = [{"title": "Bitcoin surges to new high", "source": "Test"}]
    print("Testing VADER...")
    print(analyze_headlines(test_headlines, engine='vader'))
