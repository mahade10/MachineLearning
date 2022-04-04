import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv('dhaka homeprices.csv')
x = df[['area']].to_numpy()
y = df[['price']].to_numpy()

rows = int(len(x))
print(x)
print(y)
m = 0
c = 0
l = 0.0001
itr = 200
for i in range(itr):
    ypred = m * x + c
    derivate_m = -(2 / rows) * sum(x * (y - ypred))
    derivate_c = -(2 / rows) * sum((y - ypred))
    m = m - l * derivate_m
    c = c - l * derivate_c
    print(m)

