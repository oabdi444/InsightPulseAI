import sys
import os

# Fix Python path - add parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import everything else
import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Some visualizations will be limited.")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Import our modules
try:
    from utils.data_loader import load_reviews, load_sample_data
    from utils.preprocessing import preprocess_text, clean_text_advanced
    from nlp_module.sentiment_analysis import (
        analyze_sentiment_vader, 
        analyze_sentiment_transformers,
        analyze_sentiment_rule_based,
        get_sentiment_distribution
    )
    from nlp_module.topic_modeling import extract_topics_advanced, extract_topics_simple
    from nlp_module.ner_extraction import extract_entities_advanced, format_entities
    print("✅ All modules imported successfully!")
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="InsightPulseAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .stAlert > div {
        background-color: #e8f4fd;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.markdown('<h1 class="main-header">🧠 InsightPulseAI</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced NLP Review Analyzer with Modern AI Techniques")
    
    # Sidebar for options
    st.sidebar.title("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Choose Data Source:",
        ["Sample Data", "Upload CSV", "Enter Text Manually"]
    )
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Load data based on selection
    if data_source == "Sample Data":
        if st.sidebar.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                st.session_state.df = load_sample_data()
                st.success("✅ Sample data loaded successfully!")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                if 'text' not in st.session_state.df.columns:
                    st.error("CSV must contain a 'text' column")
                    return
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
    
    elif data_source == "Enter Text Manually":
        manual_text = st.text_area("Enter reviews (one per line):", height=200)
        if st.button("Process Text"):
            if manual_text:
                texts = [line.strip() for line in manual_text.split('\n') if line.strip()]
                st.session_state.df = pd.DataFrame({'text': texts})
                st.success("✅ Text processed successfully!")
    
    # Main analysis
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Reviews", len(df))
        with col2:
            avg_length = df['text'].str.len().mean()
            st.metric("📝 Avg Length", f"{avg_length:.0f} chars")
        with col3:
            unique_reviews = df['text'].nunique()
            st.metric("🔢 Unique Reviews", unique_reviews)
        
        # Preprocessing options
        st.sidebar.subheader("🔧 Preprocessing Options")
        use_advanced_cleaning = st.sidebar.checkbox("Advanced Text Cleaning", value=True)
        remove_short_texts = st.sidebar.checkbox("Remove Short Texts (<10 chars)", value=True)
        
        # Apply preprocessing
        with st.spinner("Preprocessing text..."):
            if use_advanced_cleaning:
                df['cleaned'] = df['text'].apply(clean_text_advanced)
            else:
                df['cleaned'] = df['text'].apply(preprocess_text)
            
            if remove_short_texts:
                df = df[df['cleaned'].str.len() >= 10]
        
        # Analysis options
        st.sidebar.subheader("🎯 Analysis Options")
        analysis_type = st.sidebar.multiselect(
            "Select Analysis Types:",
            ["Sentiment Analysis", "Topic Modeling", "Named Entity Recognition", "Word Cloud"],
            default=["Sentiment Analysis", "Topic Modeling"]
        )
        
        # Sentiment Analysis
        if "Sentiment Analysis" in analysis_type:
            st.header("😊 Sentiment Analysis")
            
            sentiment_method = st.selectbox(
                "Choose Sentiment Analysis Method:",
                ["Rule-based (Fast)", "VADER (Balanced)", "Transformers (Accurate)"]
            )
            
            with st.spinner("Analyzing sentiment..."):
                if "Rule-based" in sentiment_method:
                    df['sentiment'] = df['cleaned'].apply(analyze_sentiment_rule_based)
                elif "VADER" in sentiment_method:
                    df['sentiment'] = df['cleaned'].apply(analyze_sentiment_vader)
                else:
                    df['sentiment'] = df['cleaned'].apply(analyze_sentiment_transformers)
            
            # Sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            
            if PLOTLY_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'Positive': '#2E8B57',
                            'Negative': '#DC143C',
                            'Neutral': '#FFD700'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        title="Sentiment Counts",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positive': '#2E8B57',
                            'Negative': '#DC143C',
                            'Neutral': '#FFD700'
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Fallback without plotly
                st.write("**Sentiment Distribution:**")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"- {sentiment}: {count}")
            
            # Sentiment examples
            st.subheader("📝 Sample Reviews by Sentiment")
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment in df['sentiment'].values:
                    sample_reviews = df[df['sentiment'] == sentiment]['text'].head(2)
                    st.write(f"**{sentiment} Examples:**")
                    for review in sample_reviews:
                        st.write(f"• {review}")
        
        # Topic Modeling
        if "Topic Modeling" in analysis_type:
            st.header("🏷️ Topic Modeling")
            
            n_topics = st.slider("Number of Topics", 2, 10, 5)
            
            with st.spinner("Extracting topics..."):
                try:
                    topics_result = extract_topics_advanced(df['cleaned'].tolist(), n_topics=n_topics)
                    topics, topic_words, vectorizer, lda_model = topics_result
                except:
                    # Fallback to simple method
                    topics = extract_topics_simple(df['cleaned'].tolist(), n_topics)
                    topic_words = {}
            
            # Display topics
            st.subheader("📋 Discovered Topics")
            for i, topic in enumerate(topics):
                st.write(f"**Topic {i+1}:** {topic}")
        
        # Named Entity Recognition
        if "Named Entity Recognition" in analysis_type:
            st.header("🏢 Named Entity Recognition")
            
            entity_method = st.selectbox(
                "Choose NER Method:",
                ["Pattern-based (Fast)", "Transformers (Accurate)"]
            )
            
            with st.spinner("Extracting entities..."):
                method = "transformers-based" if "Transformers" in entity_method else "pattern-based"
                df['entities'] = df['text'].apply(
                    lambda x: extract_entities_advanced(x, method=method)
                )
            
            # Aggregate all entities
            all_entities = []
            for entities_list in df['entities']:
                all_entities.extend(entities_list)
            
            if all_entities:
                try:
                    entities_df = format_entities(all_entities)
                    
                    # Entity type distribution
                    entity_counts = entities_df['label'].value_counts()
                    
                    st.subheader("📊 Entity Types Found")
                    for entity_type, count in entity_counts.head(10).items():
                        st.write(f"**{entity_type}:** {count} entities")
                    
                    # Top entities by type
                    st.subheader("🔝 Top Entities by Type")
                    for entity_type in entity_counts.head(5).index:
                        type_entities = entities_df[entities_df['label'] == entity_type]
                        top_entities = type_entities['text'].value_counts().head(3)
                        st.write(f"**{entity_type}:**")
                        for entity, count in top_entities.items():
                            st.write(f"  • {entity} ({count})")
                except:
                    st.write("Entities found but couldn't format properly. Raw entities:")
                    unique_entities = list(set([e['text'] for e in all_entities[:20]]))
                    for entity in unique_entities:
                        st.write(f"• {entity}")
            else:
                st.info("No entities found in the text.")
        
        # Word Cloud
        if "Word Cloud" in analysis_type and WORDCLOUD_AVAILABLE:
            st.header("☁️ Word Cloud")
            
            # Combine all cleaned text
            all_text = ' '.join(df['cleaned'].tolist())
            
            if all_text.strip():
                try:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis'
                    ).generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except:
                    st.info("Could not generate word cloud.")
            else:
                st.info("Not enough text to generate word cloud.")
        elif "Word Cloud" in analysis_type and not WORDCLOUD_AVAILABLE:
            st.info("Word cloud feature requires 'wordcloud' package. Install with: pip install wordcloud")
        
        # Data export
        st.header("💾 Export Results")
        
        if st.button("Download Processed Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="processed_reviews.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()