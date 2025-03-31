import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"/Users/sgajre/PycharmProjects/PythonProject/MachineLearning/dataset/MLProj.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer = imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

from sklearn.preprocessing import LabelEncoder

lebelEncoderX =  LabelEncoder()

x[:,0] = lebelEncoderX.fit_transform(x[:, 0])

print(x)

lebelEncoderY =  LabelEncoder()

y1 = lebelEncoderY.fit_transform(y)
print(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y1,test_size = 0.2, train_size = 0.8, random_state= 0)

print('X-TRAIN')
print( x_train)
print('X-TEST')
print(x_test)

print('Y-TRAIN')
print(y_train)
print('Y-TEST')
print(y_test)



