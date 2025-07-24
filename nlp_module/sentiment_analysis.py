import re
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

# VADER Sentiment (if available)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Transformers (if available)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SimpleRuleBased:
    """Simple rule-based sentiment analyzer"""
    
    def __init__(self):
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'love', 'best',
            'awesome', 'perfect', 'outstanding', 'brilliant', 'superb', 'magnificent', 'marvelous',
            'delicious', 'tasty', 'fresh', 'clean', 'friendly', 'helpful', 'polite', 'fast',
            'quick', 'efficient', 'comfortable', 'cozy', 'beautiful', 'lovely', 'nice', 'pleasant',
            'satisfied', 'happy', 'impressed', 'recommend', 'enjoy', 'favorite', 'quality'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disgusting', 'nasty',
            'disappointing', 'poor', 'cheap', 'expensive', 'overpriced', 'slow', 'rude',
            'unfriendly', 'dirty', 'unclean', 'cold', 'burnt', 'raw', 'bland', 'tasteless',
            'stale', 'old', 'uncomfortable', 'noisy', 'crowded', 'waiting', 'delayed',
            'cancelled', 'broken', 'damaged', 'defective', 'useless', 'waste', 'regret',
            'complain', 'refund', 'never', 'avoid'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'absolutely': 2.0, 'completely': 1.8,
            'totally': 1.8, 'really': 1.3, 'quite': 1.2, 'pretty': 1.1, 'so': 1.4,
            'too': 1.3, 'highly': 1.5, 'incredibly': 1.8, 'amazingly': 1.6
        }
        
        self.negations = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
    
    def analyze(self, text: str) -> str:
        """Analyze sentiment using rule-based approach"""
        if not isinstance(text, str):
            return "Neutral"
        
        words = text.lower().split()
        score = 0
        
        for i, word in enumerate(words):
            word_score = 0
            
            # Check for positive/negative words
            if word in self.positive_words:
                word_score = 1
            elif word in self.negative_words:
                word_score = -1
            
            if word_score != 0:
                # Check for intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    word_score *= self.intensifiers[words[i-1]]
                
                # Check for negations
                negation_found = False
                for j in range(max(0, i-3), i):
                    if words[j] in self.negations:
                        negation_found = True
                        break
                
                if negation_found:
                    word_score *= -1
                
                score += word_score
        
        # Classify based on score
        if score > 0.5:
            return "Positive"
        elif score < -0.5:
            return "Negative"
        else:
            return "Neutral"

# Initialize analyzers
rule_based_analyzer = SimpleRuleBased()

if VADER_AVAILABLE:
    vader_analyzer = SentimentIntensityAnalyzer()

if TRANSFORMERS_AVAILABLE:
    try:
        transformer_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
    except Exception:
        # Fallback to a more basic model
        try:
            transformer_analyzer = pipeline("sentiment-analysis")
        except Exception:
            TRANSFORMERS_AVAILABLE = False

def analyze_sentiment_rule_based(text: str) -> str:
    """Rule-based sentiment analysis (always available)"""
    return rule_based_analyzer.analyze(text)

def analyze_sentiment_vader(text: str) -> str:
    """VADER sentiment analysis (if available)"""
    if not VADER_AVAILABLE:
        return analyze_sentiment_rule_based(text)
    
    try:
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception:
        return analyze_sentiment_rule_based(text)

def analyze_sentiment_transformers(text: str) -> str:
    """Transformer-based sentiment analysis (if available)"""
    if not TRANSFORMERS_AVAILABLE:
        return analyze_sentiment_vader(text)
    
    try:
        # Truncate text if too long
        if len(text) > 512:
            text = text[:512]
        
        results = transformer_analyzer(text)
        
        if isinstance(results[0], list):
            # Handle models that return all scores
            scores = {result['label']: result['score'] for result in results[0]}
            if 'POSITIVE' in scores and scores['POSITIVE'] > 0.6:
                return "Positive"
            elif 'NEGATIVE' in scores and scores['NEGATIVE'] > 0.6:
                return "Negative"
            else:
                return "Neutral"
        else:
            # Handle models that return single prediction
            label = results[0]['label'].upper()
            score = results[0]['score']
            
            if 'POS' in label and score > 0.6:
                return "Positive"
            elif 'NEG' in label and score > 0.6:
                return "Negative"
            else:
                return "Neutral"
    
    except Exception as e:
        print(f"Transformer analysis failed: {e}")
        return analyze_sentiment_vader(text)

def get_sentiment_distribution(sentiments: List[str]) -> Dict[str, int]:
    """Get distribution of sentiments"""
    distribution = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for sentiment in sentiments:
        if sentiment in distribution:
            distribution[sentiment] += 1
    return distribution

def analyze_sentiment_batch(texts: List[str], method: str = "rule_based") -> List[str]:
    """Analyze sentiment for multiple texts"""
    if method == "vader":
        return [analyze_sentiment_vader(text) for text in texts]
    elif method == "transformers":
        return [analyze_sentiment_transformers(text) for text in texts]
    else:
        return [analyze_sentiment_rule_based(text) for text in texts]

# Default function for backward compatibility
def analyze_sentiment(text: str) -> str:
    """Default sentiment analysis function"""
    return analyze_sentiment_vader(text)