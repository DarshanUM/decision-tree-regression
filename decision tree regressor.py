import numpy as np
import pandas as p
import matplotlib.pyplot as plt

data=p.read_csv('1.csv')
x=data.iloc[:,-2].values
y=data.iloc[:,-1].values

x=np.array(x)
y=np.array(y)

x=x.reshape(-1,1)
y=y.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
dr=DecisionTreeRegressor(random_state=0)
dr.fit(x,y)

ypred=dr.predict([[4000]])



