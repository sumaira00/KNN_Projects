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

yelp=pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()
yelp['textlength']=yelp['text'].apply(len)

#EDA
sns.set_style('whitegrid')
g=sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'textlength',bins=60)

sns.boxplot(x='stars',y='textlength',data=yelp)
sns.countplot(yelp['stars'])

stars=yelp.groupby('stars').mean()
stars.corr()
sns.heatmap(stars.corr(),annot=True)

yelp_class=yelp[(yelp['stars']==1) | (yelp['stars']==5) ]
X=yelp_class['text']
y=yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train,y_train)

predictions=nb.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
        ('bow',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
        ])

X=yelp_class['text']
y=yelp_class['stars']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =101)

pipeline.fit(X_train,y_train)
predictions=pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))



























