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
df=pd.read_csv('titanic_train.csv')
df.head()
df.info()
df.describe()

#Exploratory Data Analysis
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df)
sns.countplot(x='Survived',data=df,hue='Sex')
sns.countplot(x='Survived',data=df,hue='Pclass')
sns.distplot(df['Age'].dropna(),kde=False,bins=30)
sns.countplot(x='SibSp',data=df)
df['Fare'].hist(bins=40,figsize=(10,4))

#Fill In Missing Data
sns.boxplot(x='Pclass',y='Age',data=df)
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

df['Age']=df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Drop the cabin column
df.drop('Cabin',inplace=True,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Create Dummy Variable for Sex and Embarked Columns
sex=pd.get_dummies(df['Sex'],drop_first=True)
embarked=pd.get_dummies(df['Embarked'],drop_first=True)


df=pd.concat([df,sex,embarked],axis=1)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df.drop(['PassengerId'],axis=1,inplace=True)

pclass=pd.get_dummies(df['Pclass'],drop_first=True)
df=pd.concat([df,pclass],axis=1)
df.drop(['Pclass'],axis=1,inplace=True)
 df.rename(columns={2: "SecondClass", 3: "ThirdClass"},inplace=True)
 
 
 # Importing the dataset
X = df.drop('Survived',axis=1)
y = df['Survived']

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





