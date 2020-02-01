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
df=pd.read_csv('KNN_Project_Data')
df.head()
df.info()
df.describe()


#EDA:
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(df.drop('TARGET CLASS',axis=1))
scaled_features=scalar.transform(df.drop('TARGET CLASS', axis=1))
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])

 # Importing the dataset
X = df_feat
y = df['TARGET CLASS']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)


#Call Linear Regresssion and instantiate
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)

#Predictions
predictions=model.predict(X_test)

#Evaluation of Model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


#Calculation for K value
error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))

plt.Figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',ls='dashed',marker='o',
              markerfacecolor='red',markersize=10)
plt.title('Error_Rate')

#Minimum Error Rate at K=5
#Call Linear Regresssion and instantiate
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

#Predictions
predictions=model.predict(X_test)

#Evaluation of Model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

