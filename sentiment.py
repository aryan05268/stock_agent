import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    print("VADER Lexicon not found. Downloading...")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

def analyze_headline_sentiment(headline: str) -> dict:
    """
    Analyzes a single news headline and returns sentiment scores.
    
    Returns a dictionary:
    - 'neg': Negative sentiment intensity (0 to 1)
    - 'neu': Neutral sentiment intensity (0 to 1)
    - 'pos': Positive sentiment intensity (0 to 1)
    - 'compound': Aggregated sentiment score (-1 to 1)
    """
    if not headline:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    return sia.polarity_scores(headline)

def get_average_sentiment(news_items: list) -> float:
    """
    Computes the average compound sentiment score for a list of news dictionaries.
    
    Returns:
    - Average compound score (float between -1.0 and 1.0)
    """
    if not news_items:
        return 0.0
    
    total_compound = 0.0
    valid_count = 0
    for item in news_items:
        title = item.get('title', '')
        if title:
            scores = analyze_headline_sentiment(title)
            total_compound += scores['compound']
            valid_count += 1
            
    if valid_count == 0:
        return 0.0
    return total_compound / valid_count

if __name__ == '__main__':
    # Test headlines
    headlines = [
        "Apple reports record-breaking quarterly revenue, beating Wall Street estimates.",
        "Concerns rise as regulatory crackdowns threaten tech stock gains.",
        "Microsoft announces a minor bug fix update for Office suite."
    ]
    
    print("Headline Sentiment Testing:")
    for h in headlines:
        score = analyze_headline_sentiment(h)
        print(f"\nHeadline: {h}")
        print(f"Scores: Compound={score['compound']:.4f} | Pos={score['pos']} | Neg={score['neg']}")
