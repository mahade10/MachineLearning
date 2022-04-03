import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dhaka homeprices.csv')
# print(df)
x = df[['area']]
y = df[['price']]

plt.scatter(df['area'], df['price'])
plt.xlabel('area')
plt.ylabel('price')

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30, random_state=1)
from sklearn.linear_model import LinearRegression
print(xtest,ytest)
reg = LinearRegression().fit(xtrain, ytrain)
'''
print(reg.predict([[3200]]))
print(reg.coef_)
print(reg.intercept_)
y = reg.coef_ * 3200 + reg.intercept_
'''
y = reg.predict(xtest)
print(y)