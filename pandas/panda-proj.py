import pandas as pd

store = pd.read_csv(r"../resources/Sample - Superstore_Orders.csv")
#print(data)
print(len(store))

print(store.info())
print(store.shape)
print(store.columns)
print(store.head())
print(store.tail())

