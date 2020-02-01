# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sys
import scipy

#Printing the Versions
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('sklearn: {}'.format(scipy.__version__))

#Load the dataset
data=pd.read_csv(r"C:\Users\saim-zain\Desktop\books\Python Assignments and Projects IMP\creditcard.csv")

#Exploring the dtatset
print(data.head())
print(data.columns)
print(data.shape)
print(data.describe())
print(data.corr())

data=data.sample(frac=0.1,random_state=1)

#Create a histogram of all the ratings in the average rating columns
data.hist(figsize=(20,20))

#Determine the number of fraud cases in dataset
fraud=data[data['Class']==1]
valid=data[data['Class']==0]

outlier_fraction=len(fraud)/float(len( valid))
print('Fraud Cases',len(fraud))
print('Valid Cases',len(valid))

#Correlation Matrix
fig=plt.figure(figsize=(12,9))
sns.heatmap(data.corr())

#Get All the columns from the dataframe
X = data.iloc[:, :-1].values
y = data.iloc[:, 30].values

"""
IsolationForest and LocalOutlierFactor are commonly used for outlier detection.
SVM is also used but since the data is high SVM would take time  to create a model.
The LocalOutlierFactor is an unsupervised outlier detection method and this
goes ahead and calculates the anomaly score of each sample .So it measures the
local deviation of density of a given sample with respect to its  neighbors,how
isolated the sample is with respect to its surrounding neighborhood.
Very much similar to KNearestNeighbors and we are calculation anomaly score
based on the neighbors.

Isolation Forest would return the anomaly score of each sample using isolation 
forest method,it does this by randomly selecting a feature and randomly selection a 
split value between the maximum and minimum values of the selective feature
"""
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#define the outlier detection methods
classifiers={
        "Isolation Forest": IsolationForest(max_samples=len(X),
                                                                     contamination=outlier_fraction,
                                                                     random_state=1),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                                                               contamination=outlier_fraction)
        }


#Fit the model
n_outliers=len(fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name=='Local Outlier Factor':
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
    #Reshape the prediction values to 0 for valid and 1 for fraud
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    
    n_errors=(y_pred!=y).sum()
    
    #Run Classification Matrix
    print('{}: {}'.format(clf_name,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
    
    corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
    
    



































