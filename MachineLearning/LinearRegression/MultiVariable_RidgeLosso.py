#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:23:44 2025

@author: sgajre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"/Users/sgajre/PycharmProjects/PythonProject/MachineLearning/LinearRegression/Salary_Data.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:,1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, train_size = 0.8, random_state= 0)

x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

compair = pd.DataFrame({'Actual':})


print(y_test)
print(y_pred)