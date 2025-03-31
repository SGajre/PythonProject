import pandas as pd

store = pd.read_csv(r'../resources/Sample - data.csv')

print(len(store))
df = store.dropna()
print(len(df))