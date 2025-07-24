import pandas as pd
import os

def load_reviews(path="data/sample_reviews.csv"):
    """Load reviews from CSV file"""
    try:
        if not os.path.exists(path):
            return load_sample_data()
        
        df = pd.read_csv(path)
        df.dropna(subset=["text"], inplace=True)
        df = df[df['text'].str.strip() != '']  # Remove empty strings
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return load_sample_data()

def load_sample_data():
    """Load sample review data"""
    sample_reviews = [
        "I absolutely love this restaurant! The food is amazing and the service is excellent.",
        "The service was terrible and the food was cold. Very disappointing experience.",
        "Great atmosphere and delicious food. The staff at Maria's Italian Kitchen were very friendly.",
        "Overpriced for what you get. John Smith, the manager, was rude to customers.",
        "Best pizza in New York! Located on 5th Avenue, this place is a gem.",
        "The pasta was undercooked and the sauce was bland. Not worth the $25 price.",
        "Fantastic dining experience! Chef Rodriguez really knows how to cook.",
        "Waited 45 minutes for our food. The restaurant on Main Street needs better management.",
        "Apple Inc. should consider opening a cafe here. The location is perfect.",
        "Called (555) 123-4567 to make a reservation. Staff was very helpful.",
        "The wine selection is outstanding. Contact them at info@restaurant.com for events.",
        "Visited on December 15, 2023. The Christmas decorations were beautiful.",
        "The new menu items are creative and delicious. Highly recommend the seafood.",
        "Poor hygiene standards. Health Department should inspect this place.",
        "Barack Obama ate here once according to the wall photos. Food must be good!",
        "The breakfast special for $12.99 is a great deal. Fresh ingredients daily.",
        "Microsoft employees often have lunch meetings here. Quiet and professional.",
        "Located at 123 Oak Street, this hidden gem serves the best coffee in Seattle.",
        "The online ordering system needs improvement. Website crashes frequently.",
        "Five stars! Will definitely return with my family next weekend."
    ]
    
    return pd.DataFrame({'text': sample_reviews})

def validate_data(df):
    """Validate the loaded data"""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")
    
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    
    # Remove null and empty texts
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    
    if df.empty:
        raise ValueError("No valid text data found")
    
    return df