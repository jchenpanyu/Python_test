# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 6
Page: 190-191

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
import matplotlib.pyplot as plt
 
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
