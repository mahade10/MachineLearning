import pandas as pd
import numpy as np

df = pd.read_csv('population_indian_cities.csv')
print(df)
print(df['City Name'].nunique())

#Label Encoding
from sklearn import preprocessing
model= preprocessing.LabelEncoder()
#df['City Name'] = model.fit_transform(df['City Name'])
#print(df)
#null value check
n= df['Population'].isnull().sum()
#print(n,'null values')
#one-hot encoding
dummies = pd.get_dummies(df['City Name'])
print(dummies)
#print(df)
new_df = df.drop('City Name', axis=1)
#print(new_df)
df = pd.concat([dummies,new_df], axis=1)
#print(df)
x = df.drop('Population',axis=1)
y = df[['Population']]
'''
# hashing encoding
import category_encoders as ce
encoders = ce.HashingEncoder(cols='City Name')
#df['City Name'] = encoders.fit_transform(df['City Name'])
'''
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=.2,random_state=1)
print(xtest,'\n',ytest)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xtrain,ytrain)
ypred = reg.predict(xtest)
print(ypred)
