import pandas as pd
import networkx as nx


def run_sna(file_path="facebook_combined.txt"):
    df = pd.read_csv(file_path, sep=' ', names=['user_1', 'user_2'])

    G = nx.from_pandas_edgelist(df, 'user_1', 'user_2')

    degree = dict(G.degree())

    betweenness = nx.betweenness_centrality(G, k=min(200, len(G)), seed=42)
    closeness = nx.closeness_centrality(G)

    return degree, betweenness, closeness
