import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import warnings
warnings.filterwarnings('ignore')

movies = pd.read_csv(r"Movie-Rating.csv")

print(movies.shape)
print(movies.columns)
print(len(movies.columns))

movies.columns = ['Film', 'Genre', 'CriticRating', 'AudienceRating',
       'Budget', 'Year']

print(movies.columns)
print(movies.describe)

j = sns.jointplot(data=movies, x='CriticRating', y='AudienceRating')
plt.show(j)

m1 = sns.distplot(movies.AudienceRating)
plt.show(m1)

m2 = sns.displot(movies.AudienceRating)
plt.show(m2)