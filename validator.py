import pandas as pd
from typing import List, Dict, Tuple
from analyzer import analyze_vader

def classify_sentiment(text: str) -> str:
    """Wrapper to use analyzer's VADER method for validation."""
    return analyze_vader(text)['sentiment']


def get_validation_set() -> List[Dict]:
    """
    Returns a manually labeled validation dataset of financial headlines.
    Each item contains 'text' and 'actual_sentiment' (ground truth).
    
    Labels are assigned based on the likely market impact and investor reaction.
    """
    validation_data = [
        # POSITIVE HEADLINES (Clear bullish signals)
        {"text": "Bitcoin hits new all-time high above $100,000", "actual_sentiment": "Positive"},
        {"text": "Major bank announces Bitcoin ETF approval", "actual_sentiment": "Positive"},
        {"text": "Crypto adoption surges as more retailers accept payments", "actual_sentiment": "Positive"},
        {"text": "Institutional investors pour billions into cryptocurrency", "actual_sentiment": "Positive"},
        {"text": "Ethereum upgrade successful, network faster than ever", "actual_sentiment": "Positive"},
        {"text": "Bitcoin rally continues as bulls take control", "actual_sentiment": "Positive"},
        {"text": "Analysts upgrade price target for leading cryptocurrency", "actual_sentiment": "Positive"},
        {"text": "Record breaking trading volume signals strong market interest", "actual_sentiment": "Positive"},
        {"text": "Government announces crypto-friendly regulations", "actual_sentiment": "Positive"},
        {"text": "Major tech company adds Bitcoin to balance sheet", "actual_sentiment": "Positive"},
        
        # NEGATIVE HEADLINES (Clear bearish signals)
        {"text": "Bitcoin crashes 20% amid market panic", "actual_sentiment": "Negative"},
        {"text": "Cryptocurrency exchange hacked, millions stolen", "actual_sentiment": "Negative"},
        {"text": "Regulators ban crypto trading in major economy", "actual_sentiment": "Negative"},
        {"text": "Investors flee as crypto bubble fears grow", "actual_sentiment": "Negative"},
        {"text": "Major crypto lender files for bankruptcy", "actual_sentiment": "Negative"},
        {"text": "SEC files lawsuit against cryptocurrency platform", "actual_sentiment": "Negative"},
        {"text": "Bitcoin plunges as whales dump holdings", "actual_sentiment": "Negative"},
        {"text": "Crypto scam exposed, investors lose millions", "actual_sentiment": "Negative"},
        {"text": "Warning signs emerge as leverage reaches dangerous levels", "actual_sentiment": "Negative"},
        {"text": "Market liquidations hit record high during selloff", "actual_sentiment": "Negative"},
        
        # NEUTRAL HEADLINES (Informational, no clear sentiment)
        {"text": "Bitcoin trading at $45,000 level today", "actual_sentiment": "Neutral"},
        {"text": "Cryptocurrency market cap reaches $2 trillion", "actual_sentiment": "Neutral"},
        {"text": "New report examines blockchain technology trends", "actual_sentiment": "Neutral"},
        {"text": "Crypto exchange announces new listing procedure", "actual_sentiment": "Neutral"},
        {"text": "Bitcoin mining difficulty adjusts as scheduled", "actual_sentiment": "Neutral"},
        {"text": "Analysts divided on short-term price direction", "actual_sentiment": "Neutral"},
        {"text": "Weekly trading summary shows mixed results", "actual_sentiment": "Neutral"},
        {"text": "Cryptocurrency regulation discussions continue in congress", "actual_sentiment": "Neutral"},
        {"text": "Bitcoin transaction fees remain stable this week", "actual_sentiment": "Neutral"},
        {"text": "Exchange updates trading hours for holiday", "actual_sentiment": "Neutral"},
    ]
    
    return validation_data


def evaluate_accuracy(predictions: List[str] = None, actuals: List[str] = None) -> Dict:
    """
    Evaluate the accuracy of sentiment predictions.
    
    If no predictions/actuals provided, runs evaluation on the built-in validation set.
    
    Args:
        predictions: List of predicted sentiment labels
        actuals: List of actual (ground truth) sentiment labels
        
    Returns:
        Dictionary with accuracy metrics
    """
    if predictions is None or actuals is None:
        # Run on validation set
        validation_set = get_validation_set()
        actuals = [item['actual_sentiment'] for item in validation_set]
        predictions = [classify_sentiment(item['text']) for item in validation_set]
    
    # Calculate overall accuracy
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    total = len(predictions)
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0
    
    # Calculate per-class metrics
    classes = ['Positive', 'Negative', 'Neutral']
    metrics = {}
    
    for cls in classes:
        # True Positives, False Positives, False Negatives
        tp = sum(1 for p, a in zip(predictions, actuals) if p == cls and a == cls)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == cls and a != cls)
        fn = sum(1 for p, a in zip(predictions, actuals) if p != cls and a == cls)
        
        precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': sum(1 for a in actuals if a == cls)
        }
    
    return {
        'overall_accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'per_class_metrics': metrics
    }


def generate_confusion_matrix(predictions: List[str] = None, actuals: List[str] = None) -> pd.DataFrame:
    """
    Generate a confusion matrix for sentiment predictions.
    
    Args:
        predictions: List of predicted sentiment labels
        actuals: List of actual sentiment labels
        
    Returns:
        DataFrame representing the confusion matrix
    """
    if predictions is None or actuals is None:
        validation_set = get_validation_set()
        actuals = [item['actual_sentiment'] for item in validation_set]
        predictions = [classify_sentiment(item['text']) for item in validation_set]
    
    classes = ['Positive', 'Negative', 'Neutral']
    matrix = {actual: {pred: 0 for pred in classes} for actual in classes}
    
    for pred, actual in zip(predictions, actuals):
        if actual in matrix and pred in matrix[actual]:
            matrix[actual][pred] += 1
    
    df = pd.DataFrame(matrix).T
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    
    return df


def print_evaluation_report() -> None:
    """Print a formatted evaluation report to console."""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS VALIDATION REPORT")
    print("="*60)
    
    validation_set = get_validation_set()
    
    # Show predictions vs actuals
    print(f"\nValidation Set Size: {len(validation_set)} headlines")
    print("-"*60)
    
    results = []
    for item in validation_set:
        predicted = classify_sentiment(item['text'])
        actual = item['actual_sentiment']
        match = "[OK]" if predicted == actual else "[X]"
        results.append({
            'Text': item['text'][:50] + "...",
            'Actual': actual,
            'Predicted': predicted,
            'Match': match
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Show metrics
    metrics = evaluate_accuracy()
    print("\n" + "="*60)
    print(f"OVERALL ACCURACY: {metrics['overall_accuracy']}%")
    print(f"({metrics['correct_predictions']}/{metrics['total_samples']} correct)")
    print("="*60)
    
    print("\nPer-Class Metrics:")
    print("-"*60)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*60)
    for cls, m in metrics['per_class_metrics'].items():
        print(f"{cls:<12} {m['precision']:<12} {m['recall']:<12} {m['f1_score']:<12} {m['support']:<10}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-"*60)
    cm = generate_confusion_matrix()
    print(cm)


if __name__ == "__main__":
    print_evaluation_report()
