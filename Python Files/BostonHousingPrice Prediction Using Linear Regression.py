# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sys
from pandas_datareader import data,wb
import datetime

#Printing the Versions
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))

#Basic Analysis of Data
df=pd.read_csv('USA_Housing.csv')
df.head()
df.info()
df.describe()


#Exploratory Data Analysis
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(),annot=True)

#Split The data 

# Importing the dataset
X = df.iloc[:, :-2].values
y = df.iloc[:, 5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)


#Call Linear Regresssion and instantiate
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

#Interpret the Coefficient
print(model.intercept_)
print(model.coef_)
cdf=pd.DataFrame(LR.coef_,index=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population'],columns=['Coeff'])

#Predictions
predictions=model.predict(X_test)

#Visualize the Predictions
plt.scatter(y_test,predictions)


#Plotting histogram for residuals,if its normally distributed ,model is good
sns.distplot((y_test-predictions))

#Regression Evaluation Matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE :', mean_absolute_error(y_test,predictions))
print('MSE :', mean_squared_error(y_test,predictions))
print('RMSE :', np.sqrt(mean_absolute_error(y_test,predictions)))


