def run_kmeans(dataset_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(dataset_path)
    X = df[['Age','Annual_Income_(k$)','Spending_Score']]
    X_scaled = StandardScaler().fit_transform(X)

    inertia, sil = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(10,5))
    plt.plot(inertia, marker='o')
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(sil, marker='o')
    plt.title("Silhouette Scores")
    plt.grid(True)
    plt.show()

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    print("\nCluster Profile (Mean Values):\n")
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
