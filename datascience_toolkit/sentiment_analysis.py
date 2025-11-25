import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def clean_text(t):
    t = re.sub('[^a-zA-Z]', ' ', str(t)).lower()
    words = t.split()
    stop = set(stopwords.words('english'))
    return ' '.join(w for w in words if w not in stop)


def run_sentiment_analysis(file_path="7917_1.csv"):
    nltk.download('stopwords')

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['reviews.text', 'reviews.rating'])

    df['full_review'] = df['reviews.title'].fillna("") + " " + df['reviews.text']
    df['clean'] = df['full_review'].apply(clean_text)
    df['sentiment'] = df['reviews.rating'].apply(lambda r: 'positive' if r > 4 else ('neutral' if r == 3 else 'negative'))

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    cv = CountVectorizer()
    X_train_vec = cv.fit_transform(X_train)
    X_test_vec = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, classification_report(y_test, y_pred)
