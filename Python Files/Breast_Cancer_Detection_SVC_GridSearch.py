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
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer.keys()
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
df.info()
df.describe()
 
 # Importing the dataset
X = df
y = cancer['target']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)


#Call Linear Regresssion and instantiate
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)

#Predictions
predictions=model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#Grid Search for Model Improvement
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_

grid_predictions=grid.predict(X_test)
print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))







