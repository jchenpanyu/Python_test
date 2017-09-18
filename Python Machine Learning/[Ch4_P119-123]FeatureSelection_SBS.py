# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 4
Page: 119-123
@author: vincchen
"""

"""
implement SBS algorithm in Python from scratch
"""
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
        

import pandas as pd
import numpy as np
#initial data
df_wine = pd.read_csv(r'C:\Users\vincchen\Documents\7_Coding\Python\test\scikit/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
  'Malic acid', 'Ash',
  'Alcalinity of ash', 'Magnesium',
  'Total phenols', 'Flavanoids',
  'Nonflavanoid phenols',
  'Proanthocyanins',
  'Color intensity', 'Hue',
  'OD280/OD315 of diluted wines',
  'Proline']
  

"""
A convenient way to randomly partition this dataset into a separate test and
training dataset is to use the train_test_split function from scikit-learn's
cross_validation submodule:
"""
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


"""
The min-max scaling procedure is implemented in scikit-learn and can be used as follows:
"""
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

"""
scikit-learn also implements a class for standardization
"""
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


"""
let's see our SBS implementation in action using the KNN classifier from scikit-learn:
"""
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

"""
move on to the more exciting part of our implementation and plot the classification accuracy 
of the KNN classifier that was calculated on the validation dataset.
"""
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

"""
To satisfy our own curiosity, let's see what those five features are that yielded such a 
good performance on the validation dataset:
"""
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

"""
let's evaluate the performance of the KNN classifier on the original test set:
"""
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

"""
Now let's use the selected 5-feature subset and see how well KNN performs:
"""
knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))
