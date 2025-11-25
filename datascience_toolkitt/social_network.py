def run_sna(dataset_path):
    import pandas as pd
    import networkx as nx

    df = pd.read_csv(dataset_path, sep=' ', names=['user_1','user_2'])
    G = nx.from_pandas_edgelist(df, 'user_1', 'user_2')

    degree = dict(G.degree())

    k = min(200, G.number_of_nodes())
    betweenness = nx.betweenness_centrality(G, k=k, seed=42)
    closeness = nx.closeness_centrality(G)

    print("\nTop 5 by Degree Centrality:")
    for u, s in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"User {u}: {s}")

    print("\nTop 5 by Betweenness:")
    for u, s in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"User {u}: {s:.4f}")

    print("\nTop 5 by Closeness:")
    for u, s in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"User {u}: {s:.4f}")

    return degree, betweenness, closeness
