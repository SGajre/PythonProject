import pandas as pd

df = pd.read_csv("../resources/data.csv")
print(len(df))
print(df)
print(df.shape)
print(df.columns)
print(len(df.columns))
print(type(df))
print(pd.__version__)
print(df.head())

print(df[6:])
print(df[:2])
print(df[0:200:10])

print(df.describe())
print(df.max())

print("Transpose")
print(df.describe().transpose())
df.columns = ['a','b','c','d','e']
print(df.columns)
print(df.head(1))
df.columns = ['CountryName', 'CountryCode', 'BirthRate', 'InternetUsers',
       'IncomeGroup']
print(df.columns)
print(df.head(1))

print(df[['CountryName', 'CountryCode']])

print(df.isnull())
print(df.isnull().sum())

print(df.BirthRate * df.InternetUsers)
df['myCalc'] = df['BirthRate'] * df['InternetUsers']
print(df['myCalc'])
print(df)

# axis = 0 is row and axis = 1 is column
df = df.drop('myCalc', axis=1)
print(df)

Filter = df.InternetUsers < 2

print(df[Filter])

Filter2 = df.BirthRate > 40

print(df[Filter2])

print(Filter & Filter2)

print(df[Filter & Filter2])

##print(df[ [df['InternetUsers'] < 2 ] & [df['BirthRate'] > 40]])

print(df[df.IncomeGroup == 'Low Income'])


