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
df=pd.read_csv('loan_data.csv')
df.head()
df.info()
df.describe()

#Exploratory Data Analysis
plt.figure(figsize=(10,9))
df[df['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='credit.policy=0',alpha=0.6)
df[df['credit.policy']==0]['fico'].hist(bins=35,color='red',label='credit.policy=1' , alpha=0.6)
plt.legend()

plt.figure(figsize=(10,9))
df[df['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='not.fully.paid =0',alpha=0.6)
df[df['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label='not.fully.paid =1' , alpha=0.6)
plt.legend()

plt.figure(figsize=(10,9))
sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')

sns.jointplot(x='fico',y='int.rate',data=df,color='purple')

plt.figure(figsize=(10,9))
sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',col='not.fully.paid',palette='Set1')
 
#Feautre_Engineering
df=pd.get_dummies(df,columns=['purpose'],drop_first=True)



 # Importing the dataset
X =df.drop('not.fully.paid',axis=1)
y = df['not.fully.paid']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)


#Call Model and instantiate
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

#Predictions
predictions=model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

#Predictions
predictions=model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))






