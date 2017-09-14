# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 4
Page: 108-118
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
For regularized models in scikit-learn that support L1 regularization, we can simply set the
penalty parameter to 'l1' to yield the sparse solution:
"""
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

"""
When we access the intercept terms via the lr.intercept_ attribute, we can see that the array returns three values:
Since we the fit the LogisticRegression object on a multiclass dataset, it uses the One-vs-Rest (OvR) approach by default where the first intercept belongs to the model that fits class 1 versus class 2 and 3; the second value is the intercept of the model that fits class 2 versus class 1 and 3; and the third value is the intercept of the model that fits class 3 versus class 1 and 2, respectively:
"""
print lr.intercept_

"""
Lastly, let's plot the regularization path, which is the weight coefficients of the different features for different regularization strengths:
"""
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
'magenta', 'yellow', 'black',
'pink', 'lightgreen', 'lightblue',
'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()
