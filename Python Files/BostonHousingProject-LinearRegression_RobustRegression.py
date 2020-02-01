import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

#Import the data
df=pd.read_csv('housing.data',delim_whitespace=True,header=None)

col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df.columns=col_name

sns.pairplot(df[['PTRATIO', 'B', 'LSTAT', 'MEDV']],size=1.5)

plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True)

sns.heatmap(df[['CRIM','ZN','INDUS','CHAS','MEDV']].corr(),annot=True)

X=df['RM'].values.reshape(-1,1)
y=df['MEDV'].values

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(X,y)

model.coef_
model.intercept_
sns.regplot(X,y)

sns.jointplot(x='MEDV', y='RM', data=df, kind='reg', size=10);
plt.show();

X=df['LSTAT'].values.reshape(-1,1)
model.fit(X,y)
model.coef_
model.intercept_

sns.jointplot(x='LSTAT', y='RM', data=df, kind='reg', size=10);

from sklearn.linear_model import RANSACRegressor

ransac=RANSACRegressor()

ransac.fit(X,y)

inliers=ransac.inlier_mask_
outliers=np.logical_not(inliers)

line_X=np.arange(3,40,1)
line_y_ransac=ransac.predict(line_X.reshape(-1,1))

sns.set(style='darkgrid',context='notebook')
plt.figure(figsize=(12,10))
plt.scatter(X[inliers],y[inliers],c='blue',marker='o',label='Inliers')
plt.scatter(X[outliers],y[outliers],c='brown',marker='s',label='Outliers')
plt.plot(line_X,line_y_ransac,color='red')

ransac.estimator_.coef_
ransac.estimator_.intercept_
