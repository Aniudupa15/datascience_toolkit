def run_sentiment(dataset_path):
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['reviews.text', 'reviews.rating'])

    df['full_review'] = df['reviews.title'].fillna('') + ' ' + df['reviews.text']

    df['sentiment'] = df['reviews.rating'].apply(
        lambda r: 'positive' if r >= 4 else ('neutral' if r == 3 else 'negative')
    )

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def clean(t):
        t = re.sub(r'[^a-z\s]', '', str(t).lower())
        return ' '.join([w for w in t.split() if w not in stop_words])

    df['clean'] = df['full_review'].apply(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['sentiment'], test_size=0.2,
        random_state=42, stratify=df['sentiment']
    )

    cv = CountVectorizer()
    X_train_vec = cv.fit_transform(X_train)
    X_test_vec = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred,
                          labels=['negative','neutral','positive'])

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=['negative','neutral','positive'],
                yticklabels=['negative','neutral','positive'])
    plt.title("Confusion Matrix")
    plt.show()

    return model, cv
