# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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


#Load the data
games=pd.read_csv('games.csv')
print(games.shape)
print(games.columns)

#Create a histogram of all the ratings in the average rating columns
plt.hist(games['average_rating'])

#Print the first row of all the games with zero scores
games[games['average_rating']>0].iloc[0]
games[games['average_rating']==0].iloc[0]

#Remove Any Rows without user review
games=games[games['users_rated']>0]

#Remove Any Rows with Missing Values:
games=games.dropna(axis=0)

#Draw Histogram Again
plt.hist(games['average_rating'])

#Draw a correlation matrix
fig=plt.figure(figsize=(12,9))
sns.heatmap(games.corr())

#Get all the columns from the dataframe
columns=games.columns.tolist()

#Filter the columns to remove data we dont want
columns=[c for c in columns if c not in ['id', 'type', 'name','average_rating', 'bayes_average_rating']]

#Store the variable we will be predicting on
target='average_rating'


#generate trainning and test dataset
from sklearn.model_selection import train_test_split
train=games.sample(frac=0.8,random_state=1)
test=games.loc[~games.index.isin(train.index)]

#import models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
LR=LinearRegression()

#Fit the model with traniing data
LR.fit(train[columns],train[target])

#Generate predictions for testing data
predictions=LR.predict(test[columns])

#Compute the mean squared error
mean_squared_error(predictions,test[target])

#Import the Random Forest Model
from sklearn.ensemble import RandomForestRegressor

#Initialize the model
RFR=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
#Fit the data
RFR.fit(train[columns],train[target])
#Generate predictions for testing data
prediction=RFR.predict(test[columns])

#Compute the mean squared error
mean_squared_error(prediction,test[target])

#Rating Predictions With Both the models:
rating_LR=LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR=RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

print("LR :",rating_LR)
print("RFR: ",rating_RFR)