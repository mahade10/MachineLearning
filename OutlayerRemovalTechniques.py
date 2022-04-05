import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sbn
df = pd.read_csv('weight-height.csv')
#print(df)
#print(df.isnull().sum())
#print(df['Weight'].mean())
df['Weight'].std()
posStd = df['Weight'].mean() + 2.5*df['Weight'].std()
negStd = df['Weight'].mean() - 2.5*df['Weight'].std()
originalvalues = df[(df['Weight']>=negStd) & (df['Weight']<=posStd)]
ff = originalvalues.drop(['Gender'],axis=1)
dummies = pd.get_dummies(originalvalues['Gender'],drop_first=True)
df = pd.concat([dummies,ff],axis=1)
x = df.drop(['Height'],axis=1)
y = df['Height']
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy = train_test_split(x,y,test_size=.3,random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
res = model.fit(trainx,trainy)
res = res.predict(testx)
from sklearn.metrics import r2_score
score = r2_score(testy,res)
print(score)
