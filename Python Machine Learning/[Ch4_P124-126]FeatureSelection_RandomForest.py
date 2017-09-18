# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 4
Page: 124-126
@author: vincchen
"""


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
train a forest of 10,000 trees on the Wine dataset and rank the 13 features by 
their respective importance measures
"""
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

"""
After executing the preceding code, we created a plot that ranks the different features in the Wine dataset by their relative importance; note that the feature importances are normalized so that they sum up to 1.0.
"""
import matplotlib.pyplot as plt
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
