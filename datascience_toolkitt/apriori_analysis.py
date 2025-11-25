import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(file_path="Assignment-1_Data.csv"):
    df = pd.read_csv(file_path, sep=';', on_bad_lines='skip')

    df = df.dropna(subset=['CustomerID'])
    df['BillNo'] = df['BillNo'].astype(str)
    df = df[~df['BillNo'].str.contains('C')]
    df['Itemname'] = df['Itemname'].str.strip()

    transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()

    te = TransactionEncoder()
    df_enc = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

    itemsets = apriori(df_enc, min_support=0.01, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)

    return itemsets, rules
