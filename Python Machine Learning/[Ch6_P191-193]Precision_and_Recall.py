# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 6
Page: 191-193

@author: vincchen
"""

"""
reading in the dataset directly from the UCI website using pandas:
"""
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

"""
assign the 30 features to a NumPy array X. Using LabelEncoder, we transform the class labels from 
their original string representation (M and B) into integers:
"""
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

"""
divide the dataset into a separate training dataset (80 percent of the data) and a 
separate test dataset (20 percent of the data):
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

"""
scikit-learn provides a convenient confusion_matrix function that we can use as follows:
The confusion matrix is simply a square matrix that reports the counts of the 
true positive, true negative, false positive, and false negative predictions of a classifier
"""
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
 
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, 
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

"""
Precision (PRE) and recall (REC) are performance metrics that are related to those 
true positive and true negative rates, and in fact, recall is synonymous to the true positive rate:
    PRE = TP/(TP+FP)
    REC=TPR=TP/P=TP/(FN+TP)
    F1=2*((PREXREC) / (PRE+REC))
    
A complete list of the different values that are accepted by the scoring parameter can be found at
http://scikit-learn.org/stable/modules/model_evaluation.html.
"""
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

"""
Remember that the positive class in scikit-learn is the class that is labeled as class 1. 
If we want to specify a different positive label, we can construct our own scorer via 
the make_scorer function, which we can then directly provide as 
an argument to the scoring parameter in GridSearchCV:
"""
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)
