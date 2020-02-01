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

#import uci molecular biology (promoter gene sequences) data set
url="https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
names=['Class','id','Sequence']
data=pd.read_csv(url,names=names)

#Build our dataset using a custom pandas dataframe
#each column in a dataframe is called a series
classes=data['Class']
sequences=list(data['Sequence'])
dataset={}

#Loop through the sequences and split into individual nucleotides
for i,seq in enumerate(sequences):
    
    #split into nucleatides:
    nucleotides=list(seq)
    nucleotides=[x for x in seq.strip()]
    
    nucleotides.append(classes[i])
    
    dataset[i]=nucleotides
    
dataset[0]

#turn the dataset back to dataframe
dframe=pd.DataFrame(dataset)
df=dframe.transpose()
df.rename(columns={57: 'Class'},inplace=True)
df.describe()

#record value counts for each sequence
series=[]
for name in df.columns:
    series.append(df[name].value_counts())
    
info=pd.DataFrame(series)
details=info.transpose()
details

#swtich to numerical data using pd.get_dummies()
numerical_df=pd.get_dummies(df)

#Remove  one of the class columns and rename to simply class
df=numerical_df.drop(columns=['Class_-'])

df.rename(columns={'Class_+':'Class'},inplace=True)

#import the algorithims
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn import model_selection

#define scoring method
scoring='accuracy'



#Create X and y dataset 
X = df.iloc[:, :-1].values
y = df.iloc[:, 228].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

#define models to train
names=['K Nearest Neighbors','Gaussian Process','Decision Tree','Random Forest','Neural Net','AdaBoost','Naive Bayes',
             'SVC Linear','SVC RBF','SVC Sigmoid']

classifiers=[
        KNeighborsClassifier(n_neighbors=3),
        GaussianProcessClassifier(1.0*RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        SVC(kernel='linear'),
        SVC(kernel='rbf'),
        SVC(kernel='sigmoid')
        ]

models=zip(names,classifiers)

#Evaluate Each Model in Turn
results=[]
names=[]

#Evaluate Each Model in Turn
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=1)
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
































