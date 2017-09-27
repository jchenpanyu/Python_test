# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:22:33 2017
"Python Machine Learning"
Chapter 7
Page: 233-246
    "Sentiment Analysis"

A compressed archive of the movie review dataset (84.1 MB) can be downloaded from http://ai.stanford.edu/~amaas/data/sentiment/ as a gzip-compressed tarball archive:
 
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
the CountVectorizer class takes an array of text data, which can be documents or just sentences, 
and constructs the bag-of-words model for us:
"""
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

"""
The text contains HTML markup as well as punctuation and other non-letter characters.
we will use Python's regular expression (regex) library to clean text data
"""
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

"""
Lastly, since we will make use of the cleaned text data over and over again during the next sections, let us now apply our preprocessor function to all movie reviews in our DataFrame:
"""
df['review'] = df['review'].apply(preprocessor)

"""
In the context of tokenization, another useful technique is word stemming, which is the process of transforming a word into its root form that allows us to map related words to the same stem.
"""
def tokenizer(text):
    return text.split()
# example:
tokenizer('runners like running and thus they run')

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
# example:
tokenizer_porter('runners like running and thus they run')

"""
In order to remove stop-words from the movie reviews, we will use the set of 127 English stop-words 
that is available from the NLTK library, which can be obtained by calling the nltk.download function:
"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
# example:
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

"""
we will train a logistic regression model to classify the movie reviews into positive and negative reviews.
First, we will divide the DataFrame of cleaned text documents into 25,000 documents for training and 
25,000 documents for testing:
"""
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

"""
Next we will use a GridSearchCV object to find the optimal set of parameters for our logistic regression model using 5-fold stratified cross-validation:
"""
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer,
                 tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]},
               {'vect__ngram_range': [(1,1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer,
                tokenizer_porter],
                'vect__use_idf':[False],
                'vect__norm':[None],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]}
             ]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)
"""
In the previous code example, we replaced the CountVectorizer and TfidfTransformer from the previous subsection with the TfidfVectorizer, which combines the latter transformer objects. Our param_grid consisted of two parameter dictionaries. In the first dictionary, we used the TfidfVectorizer with its default settings (use_idf=True, smooth_idf=True, and norm='l2') to calculate the tf-idfs; in the second dictionary, we set those parameters to use_idf=False, smooth_idf=False, and norm=None in order to train a model based on raw term frequencies. Furthermore, for the logistic regression classifier itself, we trained models using L2 and L1 regularization via the penalty parameter and compared different regularization strengths by defining a range of values for the inverse-regularization parameter C
"""

"""
After the grid search has finished, we can print the best parameter set:

As we can see here, we obtained the best grid search results using the regular tokenizer without Porter stemming, no stop-word library, and tf-idfs in combination with a logistic regression classifier that uses L2 regularization with the regularization strength C=10.0.
"""
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)

"""
Using the best model from this grid search, let us print the 5-fold cross-validation accuracy scores on the training set and the classification accuracy on the test dataset:
"""
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
""" The results reveal that our machine learning model can predict whether a movie review is positive or negative with 90 perc """
