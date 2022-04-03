import pandas as pd
import numpy as np
import matplotlib as plt
df = pd.read_csv('Car_Drive_Analysis_multiple_data.csv')

m = df['experience'].mean()
df.experience.fillna(m,inplace=True)
#print(df)
x = df.drop('risk',axis=1)
y = df[['risk']]


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=.2,random_state=1)
print(xtest,'\n',ytest)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xtrain,ytrain)
ypred = reg.predict(xtest)
print(ypred)
#claculate accuracy score
from sklearn.metrics import r2_score
score = r2_score(ytest,ypred)
print(score)


