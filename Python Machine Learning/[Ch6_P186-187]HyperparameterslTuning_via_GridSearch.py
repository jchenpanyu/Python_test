# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 6
Page: 186-187

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
In this section, we will take a look at a powerful hyperparameter optimization technique called grid search that can further help to improve the performance of a model by finding the optimal combination of hyperparameter values.

approach of grid search is quite simple, it's a brute-force exhaustive search paradigm where we specify a list of values for different hyperparameters, and the computer evaluates the model performance for each combination of those to obtain the optimal set:
"""
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, 
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
"""
Using the preceding code, we initialized a GridSearchCV object from the
sklearn.grid_search module to train and tune a support vector machine (SVM) pipeline. We set the param_grid parameter of GridSearchCV to a list of dictionaries to specify the parameters that we'd want to tune. For the linear SVM, we only evaluated the inverse regularization parameter C; for the RBF kernel SVM, we tuned both the C and gamma parameter. Note that the gamma parameter is specific to kernel SVMs. After we used the training data to perform the grid search, we obtained the score of the best-performing model via the best_score_ attribute and looked at its parameters, that can be accessed via the best_params_ attribute. In this particular case, the linear SVM model with 'clf__C'= 0.1' yielded the best k-fold cross-validation accuracy: 97.8 percent.
"""

"""
Finally, we will use the independent test dataset to estimate the performance of the best selected model, which is available via the best_estimator_ attribute of the GridSearchCV object:
"""
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
