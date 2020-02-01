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

#Define the instruments to download
companies_dict={
        'Amazon':'AMZN',
        'Apple':'AAPL',
        'Walgreen':'WBA',
        'Northup Gruman':'NOC',
        'Boeing':'BA',
        'Lockheed Martin':'LMT',
        'McDonalds':'MCD',
        'Intel':'INTC',
        'Navistar':'NAV',
        'IBM':'IBM',
        'Texas Instruments':'TXN',
        'Master Card':'MA',
        'Microsoft':'MSFT',
        'General Electrics':'GE',
        'Symantec':'SYMC',
        'American Express':'AXP',
        'COCA COLA':'KO',
        'Jhonson & Johnson':'JNJ',
        'Toyota':'TM',
        'Mitsubishi':'MSBHY',
        'Sony':'SNE',
        'Exxon':'XOM',
        'Chevron':'CVX',
        'Volvo Energy':'VLO',
        'Ford':'F',
        'BankOfAmrecia':'BAC'
        }

companies=sorted(companies_dict.items(),key=lambda x:x[1])

#Define which online source to use
data_source='yahoo'

#Defie the start and end date
start_date='2015-01-01'
end_date='2017-12-31'

#Use pandas_reader.data.Datareader to load desired stock data
panel_data=data.DataReader(['AMZN', 'AAPL', 'WBA', 'NOC', 'BA', 'LMT', 'MCD', 'INTC', 'NAV', 'IBM', 'TXN', 'MA', 'MSFT', 'GE', 'SYMC', 'AXP', 'KO', 'JNJ', 'TM', 'MSBHY', 'SNE', 'XOM', 'CVX', 'VLO', 'F', 'BAC'],data_source,start_date,end_date)
    
#Find the stock open and close data
stock_close=panel_data['Close']
stock_open=panel_data['Open']

print(stock_close.iloc[0])

#Calculate daily stock movement

stock_close=np.array(stock_close).T
stock_open=np.array(stock_open).T

row,col=stock_close.shape

movements=np.zeros([row,col])

for i in range(0,row):
    movements[i,:]=np.subtract(stock_close[i,:],stock_open[i,:])
    
for i in range(0,len(companies)):
    print('Company: {},Change: {}'.format(companies[i][0],sum(movements[i][:])))

#Visualization - Plot Stock Movements
plt.clf
plt.figure(figsize=(18,26))
ax1=plt.subplot(221)
plt.plot(movements[0][:])
plt.title(companies[0])

plt.subplot(222 ,sharey=ax1)
plt.plot(movements[1][:])
plt.title(companies[1])

#Import Normalizer
from sklearn.preprocessing import Normalizer
normalizer=Normalizer()
new=normalizer.fit_transform(movements)
print(new.max())
print(new.min())
print(new.mean())


#Visualization - Plot Stock Movements
plt.clf
plt.figure(figsize=(18,26))
ax1=plt.subplot(221)
plt.plot(new[0][:])
plt.title(companies[0])

plt.subplot(222 ,sharey=ax1)
plt.plot(new[1][:])
plt.title(companies[1])


#Import Necessary Libraries
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

#Define Normalizer
normalizer=Normalizer()

#Create a K-Means model - 10 cluseter
kmeans=KMeans(n_clusters=10,max_iter=1000)

#Make a pipeline chainning normalizer and KMeans
pipeline=make_pipeline(normalizer,kmeans)

#Fit pipeline to daily stock movements
pipeline.fit(movements)

#Check the inertia score
kmeans.inertia_

#Predict the cluster labels
labels=pipeline.predict(movements)

#Craete a dataframe aligning labels and companies
df=pd.DataFrame({'labels':labels,'companies':companies})

#Display df sorted by cluster label
df.sort_values('labels')

#PCA Method
from sklearn.decomposition import PCA

#Visualized the results on PCA-reduced data
reduced_data=PCA(n_components=2).fit_transform(new)

#run KMeans on reduced data
kmeans=KMeans(n_clusters=10)
kmeans.fit(reduced_data)
labels=kmeans.predict(reduced_data)


#Create a new dataframe aligning labels and companies
df=pd.DataFrame({'labels':labels,'companies':companies})

df.sort_values('labels')


#Obtain labels for each point in the mesh using our tranined model
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Define Colormap
cmap = plt.cm.Paired

plt.figure(figsize=(10, 10))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=cmap,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on Stock Market Movements (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

































