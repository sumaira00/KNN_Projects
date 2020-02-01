import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('College_data',index_col=0)
df.head()
df.info()

#EDA
sns.lmplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private',fit_reg=False,palette='coolwarm',size=6,aspect=1)
sns.lmplot(x='Outstate',y='F.Undergrad',data=df,hue='Private',fit_reg=False,size=6,aspect=1)
g=sns.FacetGrid(df,hue='Private',palette='coolwarm')
g=g.map(plt.hist,'Outstate',bins=30,alpha=.7)

g=sns.FacetGrid(df,hue='Private',palette='coolwarm')
g=g.map(plt.hist,'Grad.Rate',bins=30,alpha=.7)

#Correct the data
df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College']=100

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
model.fit(df.drop('Private',axis=1))
model.cluster_centers_
model.labels_

def converter(private):
    if private=='Yes':
        return 1
    else:
        return 0

df['Cluster']=df['Private'].apply(converter)  

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(df['Cluster'],model.labels_))
print(confusion_matrix(df['Cluster'],model.labels_))

