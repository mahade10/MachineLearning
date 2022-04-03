import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('car driving risk analysis.csv')
# print(df)
x = df[['speed']]
y = df[['risk']]
plt.scatter(x, y)
#plt.show()
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.40, random_state=1)
print(xtest,ytest)
reg = LinearRegression()

reg.fit(xtrain,ytrain)
model = reg.predict([[78]])
print(model)

