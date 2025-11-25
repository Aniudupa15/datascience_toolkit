import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def run_kmeans(file_path="Mall_Customers.csv"):
    df = pd.read_csv(file_path)
    X = df[['Age','Annual_Income_(k$)','Spending_Score']]
    X_scaled = StandardScaler().fit_transform(X)

    inertia, sil = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels))

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df, inertia, sil
