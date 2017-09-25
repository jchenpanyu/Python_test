# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 6
Page: 170-172

working with the Breast Cancer Wisconsin dataset, which contains 569 samples of malignant and benign tumor cells.
The first two columns in the dataset store the unique ID numbers of the samples and the corresponding 
diagnosis (M=malignant, B=benign), respectively. The columns 3-32 contain 30 real-value features 
that have been computed from digitized images of the cell nuclei, which can be used to build a model 
to predict whether a tumor is benign or malignant.

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
After encoding the class labels (diagnosis) in an array y, the malignant tumors are now represented as class 1, 
and the benign tumors are represented as class 0, respectively, which we can illustrate by calling
the transform method of LabelEncoder on two dummy class labels:
"""
le.transform(['M', 'B'])

"""
divide the dataset into a separate training dataset (80 percent of the data) and a 
separate test dataset (20 percent of the data):
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

"""
standardize the columns in the Breast Cancer Wisconsin dataset before we can feed them to a linear classifier
&
compress our data from the initial 30 dimensions onto a lower two-dimensional subspace via principal component analysis (PCA)

we can chain the StandardScaler, PCA, and LogisticRegression objects in a pipeline:
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()), 
                    ('pca', PCA(n_components=2)), 
                    ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

"""
In the preceding code example, we built a pipeline that consisted of two intermediate steps, a StandardScaler and a PCA transformer, and a logistic regression classifier as a final estimator. When we executed the fit method on the pipeline pipe_lr, the StandardScaler performed fit and transform on the training data, and the transformed training data was then passed onto the next object in the pipeline, the PCA. Similar to the previous step, PCA also executed fit and transform on the scaled input data and passed it to the final element of the pipeline, the estimator.
"""
