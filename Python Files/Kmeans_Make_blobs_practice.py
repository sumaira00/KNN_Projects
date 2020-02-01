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

#Import Datasets
from sklearn.datasets import make_blobs
df=make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)

#EDA
plt.scatter(df[0][:,0],df[0][:,1],c=df[1],cmap='rainbow')


#Model
from sklearn.cluster import KMeans
model=KMeans(n_clusters=4)
model.fit(df[0])
model.cluster_centers_
model.labels_

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(df[0][:,0],df[0][:,1],c=model.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(df[0][:,0],df[0][:,1],c=df[1],cmap='rainbow')