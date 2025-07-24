import re
import string
from typing import List, Optional

def preprocess_text(text: str) -> str:
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def clean_text_advanced(text: str) -> str:
    """Advanced text preprocessing with multiple cleaning steps"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', '', text)
    
    # Remove extra punctuation but keep sentence structure
    text = re.sub(r'[{}]'.format(re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')), ' ', text)
    
    # Remove numbers but keep text
    text = re.sub(r'\\b\\d+\\b', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text: str, custom_stopwords: Optional[List[str]] = None) -> str:
    """Remove common English stopwords"""
    # Basic English stopwords
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once'
    }
    
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    return ' '.join(filtered_words)

def extract_key_phrases(text: str, min_length: int = 2) -> List[str]:
    """Extract key phrases from text"""
    # Simple phrase extraction based on capitalization and length
    phrases = []
    
    # Find phrases with multiple capitalized words
    capitalized_phrases = re.findall(r'\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+\\b', text)
    phrases.extend(capitalized_phrases)
    
    # Find quoted phrases
    quoted_phrases = re.findall(r'\"([^\"]+)\"', text)
    phrases.extend(quoted_phrases)
    
    # Filter by minimum length
    phrases = [phrase.strip() for phrase in phrases if len(phrase.split()) >= min_length]
    
    return list(set(phrases))  # Remove duplicates

def text_statistics(text: str) -> dict:
    """Calculate various text statistics"""
    if not isinstance(text, str):
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    return {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'unique_words': len(set(word.lower() for word in words))
    }