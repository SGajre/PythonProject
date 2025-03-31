import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("../resources/data.csv")

print(df)
vis1 = sns.displot(df['InternetUsers'])
plt.show(vis1)