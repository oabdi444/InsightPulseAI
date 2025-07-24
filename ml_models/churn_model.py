from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(df):
    df['label'] = df['rating'].apply(lambda x: 1 if x <= 2 else 0)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english')
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_vec, y_train)
    print(classification_report(y_test, clf.predict(X_test_vec)))

    return clf, tfidf
