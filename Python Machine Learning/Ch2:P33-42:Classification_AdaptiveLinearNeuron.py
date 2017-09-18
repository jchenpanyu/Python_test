# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017

"Python Machine Learning"
Chapter 2
Page: 33-42


@author: vincchen
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# initialize data
df = pd.read_csv(r'C:\Users\vincchen\Documents\7_Coding\Python\test\data\iris.data')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values


# Let us now plot the cost against the number of epochs for the two different learning rates:
# fig, ax = plt.subplots(nrows=1, ncols=2, )

fig1 = plt.figure(figsize=(8, 4))
ax = plt.subplot(121)
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax.plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax.set_xlabel('Epochs')
ax.set_ylabel('log(Sum-squared-error)')
ax.set_title('Adaline - Learning rate 0.01')

ax = plt.subplot(122)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax.plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax.set_xlabel('Epochs')
ax.set_ylabel('Sum-squared-error')
ax.set_title('Adaline - Learning rate 0.0001')


"""
Standardization the data:
to standardize the jth feature, we simply need to subtract the sample mean(j)
from every training sample and divide it by its standard deviation(j) :
"""
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
# After standardization, we will train the Adaline again and see that it now converges
# using a learning rate 0.01:
ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

fig2 = plt.figure(figsize=(8, 4))
ax2 = plt.subplot(121)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

ax2 = plt.subplot(122)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.show()
