import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import networkx as nx


# -------------------------------------------------------------
# 1. K-MEANS CLUSTERING
# -------------------------------------------------------------
def run_kmeans(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['Age','Annual_Income_(k$)','Spending_Score']]
    X_scaled = StandardScaler().fit_transform(X)

    inertia, sil = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(2,11), inertia, marker='o')
    plt.title("Elbow Method")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(range(2,11), sil, marker='o')
    plt.title("Silhouette Scores")
    plt.grid(True)
    plt.show()

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    print("\nCluster Profile:\n")
    print(df.groupby('Cluster')[['Age','Annual_Income_(k$)','Spending_Score']].mean())

    sns.scatterplot(x=df['Annual_Income_(k$)'], y=df['Spending_Score'],
                    hue=df['Cluster'], palette='viridis', s=80)
    plt.title("Customer Segments")
    plt.grid(True)
    plt.show()

    sns.scatterplot(x=X_scaled[:,1], y=X_scaled[:,2],
                    hue=df['Cluster'], palette='viridis', s=80)
    plt.title("Scaled Clusters")
    plt.xlabel("Scaled Income")
    plt.ylabel("Scaled Score")
    plt.grid(True)
    plt.show()

    return df


# -------------------------------------------------------------
# 2. APRIORI
# -------------------------------------------------------------
def run_apriori(dataset_path):
    df = pd.read_csv(dataset_path, sep=';', on_bad_lines='skip')
    df = df.dropna(subset=['CustomerID'])
    df['BillNo'] = df['BillNo'].astype(str)
    df = df[~df['BillNo'].str.contains('C')]
    df['Itemname'] = df['Itemname'].str.strip()

    transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()

    te = TransactionEncoder()
    df_enc = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

    itemsets = apriori(df_enc, min_support=0.01, use_colnames=True)
    print("\nFrequent Itemsets:\n", itemsets)

    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
    print("\nAssociation Rules:\n", rules)

    return itemsets, rules


# -------------------------------------------------------------
# 3. SENTIMENT ANALYSIS
# -------------------------------------------------------------
def clean_text(t):
    t = re.sub('[^a-zA-Z]', ' ', str(t)).lower()
    words = t.split()
    stop = set(nltk.corpus.stopwords.words('english'))
    return " ".join(w for w in words if w not in stop)


def run_sentiment(dataset_path):
    nltk.download('stopwords')

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['reviews.text', 'reviews.rating'])

    df['full_review'] = df['reviews.title'].fillna("") + " " + df['reviews.text']
    df['clean'] = df['full_review'].apply(clean_text)
    df['sentiment'] = df['reviews.rating'].apply(
        lambda r: 'positive' if r > 4 else ('neutral' if r == 3 else 'negative')
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    cv = CountVectorizer()
    X_train_vec = cv.fit_transform(X_train)
    X_test_vec = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, cv


# -------------------------------------------------------------
# 4. SOCIAL NETWORK ANALYSIS
# -------------------------------------------------------------
def run_sna(dataset_path):
    df = pd.read_csv(dataset_path, sep=' ', names=['user_1', 'user_2'])
    G = nx.from_pandas_edgelist(df, 'user_1', 'user_2')

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, k=min(200, len(G)), seed=42)
    closeness = nx.closeness_centrality(G)

    print("\nTop 5 by Degree:", sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5])
    print("\nTop 5 by Betweenness:", sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5])
    print("\nTop 5 by Closeness:", sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5])

    return degree, betweenness, closeness


# -------------------------------------------------------------
# 5. RUN ALL
# -------------------------------------------------------------
def run_all(kmeans_file, apriori_file, sentiment_file, sna_file):
    print("\n=== K-MEANS CLUSTERING ===")
    run_kmeans(kmeans_file)

    print("\n=== APRIORI ===")
    run_apriori(apriori_file)

    print("\n=== SENTIMENT ANALYSIS ===")
    run_sentiment(sentiment_file)

    print("\n=== SOCIAL NETWORK ANALYSIS ===")
    run_sna(sna_file)

    print("\nALL ANALYSIS COMPLETED.")
