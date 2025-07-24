import re
from typing import List, Tuple, Dict, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class SimpleTopicExtractor:
    """Simple topic extraction using word frequency and patterns"""
    
    def __init__(self):
        self.stopwords = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'was', 'it', 'in', 'for',
            'with', 'he', 'be', 'of', 'his', 'her', 'they', 'them', 'their', 'have', 'had',
            'has', 'having', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'i', 'you', 'we', 'she', 'him', 'us', 'me', 'my', 'your',
            'our', 'this', 'that', 'these', 'those', 'an', 'as', 'are', 'but', 'or', 'if',
            'not', 'no', 'so', 'by', 'from', 'up', 'out', 'down', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'now'
        }
    
    def extract_topics(self, texts: List[str], n_topics: int = 3) -> List[str]:
        """Extract topics using simple word frequency analysis"""
        if not texts:
            return []
        
        # Combine all texts and extract words
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            filtered_words = [word for word in words if word not in self.stopwords]
            all_words.extend(filtered_words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Group related words into topics
        topics = []
        used_words = set()
        
        for i in range(n_topics):
            topic_words = []
            for word, count in word_counts.most_common():
                if word not in used_words and len(topic_words) < 5:
                    topic_words.append(word)
                    used_words.add(word)
            
            if topic_words:
                topics.append(f"Topic {i+1}: " + ", ".join(topic_words))
        
        return topics

# Initialize simple extractor
simple_extractor = SimpleTopicExtractor()

def extract_topics_simple(texts: List[str], n_topics: int = 3) -> List[str]:
    """Simple topic extraction (always available)"""
    return simple_extractor.extract_topics(texts, n_topics)

def extract_topics_advanced(texts: List[str], n_topics: int = 5) -> Tuple[List[str], Dict[str, Any], Any, Any]:
    """Advanced topic modeling using LDA (if sklearn available)"""
    if not SKLEARN_AVAILABLE or not texts:
        simple_topics = extract_topics_simple(texts, n_topics)
        return simple_topics, {}, None, None
    
    try:
        # Clean and filter texts
        cleaned_texts = []
        for text in texts:
            if isinstance(text, str) and len(text.strip()) > 10:
                cleaned_texts.append(text.strip())
        
        if len(cleaned_texts) < 2:
            return extract_topics_simple(texts, n_topics), {}, None, None
        
        # Vectorize texts
        vectorizer = CountVectorizer(
            max_df=0.85,
            min_df=2,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b',
            max_features=1000
        )
        
        doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
        
        if doc_term_matrix.shape[1] < n_topics:
            return extract_topics_simple(texts, n_topics), {}, None, None
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online',
            learning_offset=50.0
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        topic_words = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-8:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words[f"Topic {topic_idx + 1}"] = top_words
            topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words[:5]))
        
        return topics, topic_words, vectorizer, lda
    
    except Exception as e:
        print(f"Advanced topic modeling failed: {e}")
        return extract_topics_simple(texts, n_topics), {}, None, None

def visualize_topics(topic_words: Dict[str, List[str]]):
    """Create visualization for topics"""
    if not PLOTLY_AVAILABLE or not topic_words:
        return None
    
    try:
        fig = go.Figure()
        
        for i, (topic_name, words) in enumerate(topic_words.items()):
            fig.add_trace(go.Bar(
                name=topic_name,
                x=words[:5],
                y=[5-j for j in range(5)],
                text=words[:5],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Topic Words Distribution",
            xaxis_title="Words",
            yaxis_title="Importance",
            barmode='group',
            height=400
        )
        
        return fig
    
    except Exception:
        return None

def extract_topics_nmf(texts: List[str], n_topics: int = 5) -> List[str]:
    """Extract topics using Non-negative Matrix Factorization"""
    if not SKLEARN_AVAILABLE:
        return extract_topics_simple(texts, n_topics)
    
    try:
        # Use TF-IDF for NMF
        vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=2,
            stop_words='english',
            lowercase=True,
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Fit NMF model
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=100)
        nmf.fit(tfidf_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
        
        return topics
    
    except Exception:
        return extract_topics_simple(texts, n_topics)

# Default function for backward compatibility
def extract_topics(texts: List[str], n_topics: int = 3) -> List[str]:
    """Default topic extraction function"""
    topics, _, _, _ = extract_topics_advanced(texts, n_topics)
    return topics