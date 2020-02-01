# Importing the libraries
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import sklearn
import sys

#Printing the Versions
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))

#Importing the Libraries
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd


#Loading the dataset
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names=['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion',
           'single_epithelial_size','bare_nuclei','bland_chromatin','normal_nuclei',
           'mitosis','class']
df=pd.read_csv(url,names=names)

#Pre-Process the data
df.replace('?',-99999,inplace=True)
df.drop(['id'],axis=1,inplace=True)
df.axes


#Do Dataset Visualization
df.describe()

#Plot Histogram For Each Feature
df.hist(figsize=(10,10))

#Create SCatter Plot Matrix
scatter_matrix(df,figsize=(10,10))


#Create X and Y datasets for trainning
X = df.iloc[:, :-1].values
y = df.iloc[:, 9].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Define models to train
models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

#Evaluate Each Model in Turn
results=[]
names=[]

for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=8)
    cv_results = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print((name,"{0:0.4f}".format(cv_results.mean()),"{0:0.4f}".format(cv_results.std())))
    
#Make Predictions on Validation dataset
for name,model in models:
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    print(name)
    print(accuracy_score(y_test,predictions))
    print(classification_report(y_test,predictions))
    
example=np.array([[4,2,1,1,1,2,3,2,4]])
