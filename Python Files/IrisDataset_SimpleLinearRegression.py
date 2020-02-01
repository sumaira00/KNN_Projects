import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

df = pd.read_csv('iris.data',header=-1,names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

x=10*np.random.randn(100)
y=3 * x  + np.random.randn(100)

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

model=LinearRegression(fit_intercept=True)

X = x.reshape(-1,1)

model.fit(X,y)

model.coef_
model.intercept_

x_fit = np.linspace(-1,11)
X_fit=x_fit.reshape(-1,1)

y_fit=model.predict(X_fit)

plt.scatter(x,y)
plt.plot(x_fit,y_fit)