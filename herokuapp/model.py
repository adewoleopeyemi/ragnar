# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:26:04 2020

@author: adewole opeyemi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


dataset=pd.read_csv('datasets_84803_196262_BankNote_Authentication.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model=pickle.load(open('model.pkl', 'rb'))

print(model.predict([[1, 2, 3, 4]]))


