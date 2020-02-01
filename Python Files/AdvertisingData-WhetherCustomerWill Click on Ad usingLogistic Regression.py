# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sys
#from pandas_datareader import data,wb
import datetime

#Printing the Versions
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))

#Basic Analysis of Data
df=pd.read_csv(r"C:\Users\saim-zain\Desktop\books\Python Assignments and Projects IMP\advertising.csv")
df.head()
df.info()
df.describe()

#Exploratory Data Analysis
df['Age'].hist(bins=40,figsize=(10,4))
sns.jointplot(x='Age',y='Area Income',data=df)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=df,kind='kde')
sns.jointplot(x='Daily Internet Usage',y='Daily Time Spent on Site',data=df)
sns.pairplot(df,hue='Clicked on Ad')
 
 # Importing the dataset
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)


#Call Linear Regresssion and instantiate
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

#Predictions
predictions=model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))





