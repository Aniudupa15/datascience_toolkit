def run_apriori(dataset_path):
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    df = pd.read_csv(dataset_path, sep=';', on_bad_lines='skip')
    df = df.dropna(subset=['CustomerID'])
    df['BillNo'] = df['BillNo'].astype(str)
    df = df[~df['BillNo'].str.contains('C')]
    df['Itemname'] = df['Itemname'].str.strip()

    transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()

    te = TransactionEncoder()
    df_enc = pd.DataFrame(te.fit(transactions).transform(transactions),
                          columns=te.columns_)

    itemsets = apriori(df_enc, min_support=0.01, use_colnames=True)
    print("\nFrequent Itemsets:\n", itemsets)

    rules = association_rules(itemsets, metric="confidence",
                              min_threshold=0.5)
    print("\nAssociation Rules:\n", rules)

    return itemsets, rules
