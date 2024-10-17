import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

try:
    df = pd.read_csv('associationmining.csv')
except FileNotFoundError:
    print("Error: The file 'transactions.csv' was not found.")
    exit()

print("DataFrame preview:")
print(df.head())

basket = df.groupby(['TransactionId', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('TransactionId')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)  

print("\nOne-hot encoded DataFrame:")
print(basket.head())

frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

print("\nFrequent itemsets:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("\nAssociation rules:")
print(rules)

for index, row in rules.iterrows():
    print(f"Rule: {row['antecedents']} -> {row['consequents']}, Support: {row['support']}, Confidence: {row['confidence']}, Lift: {row['lift']}")
