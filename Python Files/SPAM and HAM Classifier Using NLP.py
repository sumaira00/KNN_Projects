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
messages=[lines.rstrip() for lines in open('smsspamcollection/SMSSpamCollection')]
for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
messages=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names
                     =['label','message'])

messages.describe()
messages.groupby('label').describe()
messages['length']=messages['message'].apply(len)

messages['length'].plot.hist(bins=150)
messages['length'].describe()
messages[messages['length']==910]['message'].iloc[0]
messages.hist(column='length',by='label',bins=60,figsize=(12,4))

import string
mess='Sample message! Notice: it has punctuation.'
string.punctuation
nopunc=[c for c in mess if c not in string.punctuation]
from nltk.corpus import stopwords
stopwords.words('english')
nopunc=''.join(nopunc)
nopunc.split()
clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def text_process(mess):
    '''
    1.Remove Punctuation
    2.remove stop words
    3.return list of clean text words
    '''
    
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
messages['message'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
mess4=messages['message'][3]
print(mess4)
bow4=bow_transformer.transform([mess4])
print(bow4)
bow_transformer.get_feature_names()[4068]
bow4.shape
messages_bow=bow_transformer.transform(messages['message'])
messages_bow.nnz
sparsity=(100.0*messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('Sparsity : {}'.format((sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)
tfidf4=tfidf_transformer.transform(bow4)
print(tfidf4)
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf=tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(messages_tfidf,messages['label'])
spam_detect_model.predict(tfidf4)[0]
messages['label'][3]
all_pred=spam_detect_model.predict(messages_tfidf)

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test=train_test_split(messages['message'],messages['label'],test_size=0.3)


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
        ])

pipeline.fit(msg_train,label_train)
predictions=pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))
