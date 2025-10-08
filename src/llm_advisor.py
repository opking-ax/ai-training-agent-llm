import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def get_sentiment(new_headline: str) -> float:
    """
    Get a sentimental analysis for a given headline
    Arg:
        new_headline (str): a string contains the news headline
    Returns:
        A score based on the headline [postive: +1, neutral: 0, negative: -1]
    """
    if not new_headline or new_headline.strip == "":
        return 0.0
    
    score = sid.polarity_scores(new_headline)['compound']

    if score >= 0.05:
        return 1.0
    elif score <= -0.05:
        return -1.0
    else:
        return 0.0