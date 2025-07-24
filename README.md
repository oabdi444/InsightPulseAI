# InsightPulseAI - Advanced NLP Review Analyzer 🧠

A modern, production-ready NLP application for analyzing customer reviews with multiple AI techniques and interactive visualizations.

## Features 

- **Multiple Sentiment Analysis Methods**: Rule-based, VADER, and Transformer-based
- **Advanced Topic Modeling**: LDA and NMF algorithms with interactive visualizations  
- **Smart Named Entity Recognition**: Pattern-based and Transformer-based entity extraction
- **Interactive Dashboard**: Built with Streamlit for easy exploration
- **Flexible Data Input**: Sample data, CSV upload, or manual text entry
- **Rich Visualizations**: Plotly charts, word clouds, and entity distributions
- **Production Ready**: Robust error handling and fallback mechanisms

## Quick Start 

### 1. Clone or Create Project Structure
```bash
mkdir InsightPulseAI
cd InsightPulseAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app/main_app.py
```

### 4. Open Your Browser
Navigate to `http://localhost:8501` to use the application.

## Project Structure 

```
InsightPulseAI/
├── app/
│   └── main_app.py              # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   └── preprocessing.py         # Text preprocessing functions
├── nlp_module/
│   ├── __init__.py
│   ├── sentiment_analysis.py    # Multiple sentiment analysis methods
│   ├── topic_modeling.py        # Advanced topic extraction
│   └── ner_extraction.py        # Named entity recognition
├── data/
│   └── sample_reviews.csv       # Sample dataset
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Key Improvements Over Basic spaCy Implementation 

### 1. **Multiple Analysis Methods**
- **Sentiment**: Rule-based → VADER → Transformers (with fallbacks)
- **NER**: Pattern-based → Transformer-based (with fallbacks)  
- **Topics**: Simple frequency → LDA → NMF algorithms

### 2. **Production-Ready Architecture**
- Graceful degradation when libraries aren't available
- Comprehensive error handling
- Modular design for easy maintenance
- Proper Python packaging

### 3. **Enhanced User Experience**
- Interactive parameter tuning
- Multiple data input methods
- Rich visualizations with Plotly
- Real-time processing feedback

### 4. **Advanced NLP Techniques**
- Transformer models (BERT-based) for high accuracy
- Custom pattern matching for domain-specific entities
- Advanced text preprocessing pipeline
- Ensemble methods for improved results

## Usage Examples 

### Analyze Sample Data
```python
from utils.data_loader import load_sample_data
from nlp_module.sentiment_analysis import analyze_sentiment_transformers

# Load data
df = load_sample_data()

# Analyze sentiment
df['sentiment'] = df['text'].apply(analyze_sentiment_transformers)
print(df['sentiment'].value_counts())
```

### Extract Entities
```python
from nlp_module.ner_extraction import extract_entities_advanced

text = "I visited Maria's Italian Kitchen on 5th Avenue and met John Smith."
entities = extract_entities_advanced(text, method="transformers-based")

for entity in entities:
    print(f"{entity['text']} -> {entity['label']} ({entity['confidence']:.2f})")
```

### Topic Modeling
```python
from nlp_module.topic_modeling import extract_topics_advanced

reviews = ["Great food and service", "Terrible experience", "Amazing restaurant"]
topics, topic_words, _, _ = extract_topics_advanced(reviews, n_topics=2)

for topic in topics:
    print(topic)
```

## Configuration Options 

### Sentiment Analysis
- **Rule-based**: Fast, no dependencies
- **VADER**: Balanced speed and accuracy
- **Transformers**: Highest accuracy, requires more resources

### Topic Modeling  
- **Simple**: Word frequency-based
- **LDA**: Latent Dirichlet Allocation
- **NMF**: Non-negative Matrix Factorization

### Named Entity Recognition
- **Pattern-based**: Custom rules for domain-specific entities
- **Transformers**: Pre-trained BERT models for general entities

## Dependencies 

### Core Requirements
- `streamlit` - Web interface
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `plotly` - Interactive visualizations

### Optional (with fallbacks)
- `transformers` + `torch` - Advanced NLP models
- `vaderSentiment` - Sentiment analysis
- `wordcloud` - Word cloud generation

## Performance Tips 

1. **For Speed**: Use rule-based and VADER methods
2. **For Accuracy**: Use transformer-based methods
3. **For Large Datasets**: Process in batches
4. **For Production**: Use caching and async processing

## Troubleshooting 🔧

### Common Issues
- **Import Errors**: Install optional dependencies or use fallback methods
- **Memory Issues**: Reduce batch size or use lighter models  
- **Slow Processing**: Switch to faster analysis methods
- **Path Issues**: Run from project root directory

### Getting Help
- Check the console for detailed error messages
- Verify all files are in correct locations
- Ensure Python 3.8+ is being used

## Contributing 

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License 

MIT License - see LICENSE file for details.

## Acknowledgments 

- Hugging Face for transformer models
- Streamlit for the web framework
- scikit-learn for machine learning algorithms
- The open-source NLP community
"""
