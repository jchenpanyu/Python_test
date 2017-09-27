# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:22:33 2017
"Python Machine Learning"
Chapter 7
Page: 233-246
    "Sentiment Analysis"
@author: vincchen
"""

"""
The movie review dataset consists of 50,000 polar movie reviews that are labeled as either positive or negative;
here, positive means that a movie was rated with more than six stars on IMDb, 
and negative means that a movie was rated with fewer than five stars on IMDb.
"""

import pandas as pd
import os
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path ='./aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
df.columns = ['review', 'sentiment']

"""
Since the class labels in the assembled dataset are sorted, we will now shuffle DataFrame using 
the permutation function from the np.random submodule—this will be useful to split the dataset into 
training and test sets in later sections when we will stream the data from our local drive directly.
"""
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)

"""

In this section, we will make use of the partial_fit function of the SGDClassifier in scikit-learn 
to stream the documents directly from our local drive and train a logistic regression model using 
small minibatches of documents.

"""

"""
define a tokenizer function that cleans the unprocessed text data from our movie_data.csv file
"""
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

"""
Next we define a generator function, stream_docs, that reads in and returns one document at a time:
"""
def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

"""            
To verify that our stream_docs function works correctly, let us read in the first document from 
the movie_data.csv file, which should
"""         
next(stream_docs(path='./movie_data.csv'))

"""
We will now define a function, get_minibatch, that will take a document stream from the stream_docs function 
and return a particular number of documents specified by the size parameter:
"""
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

"""
makes use of the Hashing trick via the 32-bit MurmurHash3 algorithm by Austin Appleby (
"""
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

"""
Now comes the really interesting part. Having set up all the complementary functions, we can now start the out-of-core learning using the following code:
"""
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
"""
We initialized the progress bar object with 45 iterations and, in the following for loop, we iterated over 45 minibatches of documents where each minibatch consists of 1,000 documents each.
"""

"""
Having completed the incremental learning process, we will use the last 5,000 documents to 
evaluate the performance of our model:
"""
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

"""
As we can see, the accuracy of the model is 87 percent, slightly below the accuracy that we achieved 
in the previous section using the grid search for hyperparameter tuning. However, out-of-core learning 
is very memory-efficient and took less than a minute to complete. Finally, we can use the last 5,000
documents to update our model:
"""
clf = clf.partial_fit(X_test, y_test)
